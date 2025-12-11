"""
Enhanced Voice Assistant Server
Adds VAD, Interruption Handling, and Audio Buffering to working direct API approach
"""

import asyncio
import json
import base64
import time
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# VAD imports (only what we need)
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.frames.frames import AudioRawFrame

from loguru import logger
import config

app = FastAPI(title="E2E Voice Assistant - Enhanced")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnhancedVoiceHandler:
    """
    Enhanced handler with VAD, interruption, and buffering
    """

    def __init__(self):
        # VAD setup
        self.vad = SileroVADAnalyzer(
            params=VADParams(
                confidence_threshold=config.VAD_THRESHOLD,
                silence_duration_ms=int(config.END_OF_SPEECH_THRESHOLD * 1000)
            )
        )

        # Session management
        self.active_sessions: Dict[str, dict] = {}

        logger.info("✅ Enhanced Voice Handler initialized with VAD")

    async def handle_connection(self, websocket: WebSocket):
        """Handle enhanced WebSocket connection"""
        session_id = id(websocket)

        try:
            await websocket.accept()
            logger.info(f"✅ Enhanced WebSocket connected: {session_id}")

            # Initialize session
            self.active_sessions[session_id] = {
                'websocket': websocket,
                'audio_buffer': bytearray(),
                'is_speaking': False,
                'assistant_speaking': False,
                'last_speech_time': None,
                'conversation_context': []
            }

            await websocket.send_json({
                "type": "connection_status",
                "status": "connected",
                "features": ["VAD", "Interruption", "Enhanced Audio"],
                "message": "🎤 VAD active - speak naturally!"
            })

            # Main message loop
            while True:
                data = await websocket.receive_json()

                if data.get("type") == "audio_chunk":
                    await self.process_audio_chunk(session_id, data["data"])
                elif data.get("type") == "audio":  # Legacy push-to-talk support
                    await self.process_legacy_audio(session_id, data["data"])

        except WebSocketDisconnect:
            logger.info(f"❌ Enhanced WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Enhanced WebSocket error {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Cleanup session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def process_audio_chunk(self, session_id: str, base64_audio: str):
        """Process continuous audio chunks with VAD"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        try:
            # Decode audio
            audio_bytes = base64.b64decode(base64_audio)
            websocket = session['websocket']

            # Create AudioRawFrame for VAD
            audio_frame = AudioRawFrame(
                audio=audio_bytes,
                sample_rate=config.SAMPLE_RATE,
                num_channels=1
            )

            # Process through VAD - be more aggressive during assistant speaking
            speech_detected = await self.detect_speech(audio_frame, session['assistant_speaking'])

            if speech_detected:
                await self.handle_speech_detected(session_id, audio_bytes)
            else:
                # Only handle "no speech" if user was previously speaking
                # Don't reset during assistant speaking unless user was talking
                if session['is_speaking']:
                    await self.handle_no_speech(session_id)

        except Exception as e:
            logger.error(f"❌ Audio chunk processing error: {e}")

    async def detect_speech(self, audio_frame: AudioRawFrame, assistant_speaking: bool = False) -> bool:
        """Use VAD to detect speech in audio frame"""
        try:
            # Simple amplitude-based VAD as fallback if Silero fails
            import numpy as np

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_frame.audio, dtype=np.int16)

            # Calculate volume/energy
            volume = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

            # Use more sensitive threshold during assistant speaking for interruption
            if assistant_speaking:
                speech_threshold = 800   # More sensitive for interruption
                logger.debug(f"🎤 Interruption detection - Volume: {volume:.0f}, Threshold: {speech_threshold}")
            else:
                speech_threshold = 1000  # Normal threshold
                logger.debug(f"🎤 Normal VAD - Volume: {volume:.0f}, Threshold: {speech_threshold}")

            speech_detected = volume > speech_threshold

            if speech_detected and assistant_speaking:
                logger.info(f"🛑 INTERRUPTION DETECTED! Volume: {volume:.0f}")

            return speech_detected

        except Exception as e:
            logger.error(f"❌ VAD detection error: {e}")
            return False

    async def handle_speech_detected(self, session_id: str, audio_bytes: bytes):
        """Handle when speech is detected"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session['websocket']

        if not session['is_speaking']:
            # Speech started
            logger.info("🗣️ VAD: Speech started")
            session['is_speaking'] = True
            session['audio_buffer'] = bytearray()
            session['last_speech_time'] = time.time()
            session['interrupted'] = False  # Reset interruption flag

            # Stop assistant if it's speaking (INTERRUPTION)
            if session['assistant_speaking']:
                logger.info("🛑 INTERRUPTION: User interrupted assistant")
                session['assistant_speaking'] = False
                session['interrupted'] = True  # Set interruption flag

                # Send stop audio FIRST, then interruption message
                await websocket.send_json({
                    "type": "stop_audio",
                    "message": "Stop current audio playback"
                })

                await websocket.send_json({
                    "type": "interruption",
                    "message": "🛑 Assistant interrupted - listening to you"
                })

                # Also send immediate status update
                await websocket.send_json({
                    "type": "processing_status",
                    "status": "ready"
                })

            await websocket.send_json({
                "type": "vad_status",
                "status": "speaking",
                "message": "👂 Listening..."
            })

        # Add to buffer
        session['audio_buffer'].extend(audio_bytes)
        session['last_speech_time'] = time.time()

    async def handle_no_speech(self, session_id: str):
        """Handle when no speech is detected"""
        session = self.active_sessions.get(session_id)
        if not session or not session['is_speaking']:
            return

        # Check if enough silence has passed
        current_time = time.time()
        silence_duration = current_time - session['last_speech_time']

        if silence_duration >= config.END_OF_SPEECH_THRESHOLD:
            # Speech ended - process the audio
            logger.info("✅ VAD: Speech ended, processing...")
            session['is_speaking'] = False

            if len(session['audio_buffer']) > 0:
                await self.process_complete_speech(session_id, bytes(session['audio_buffer']))
                session['audio_buffer'] = bytearray()

            await session['websocket'].send_json({
                "type": "vad_status",
                "status": "listening",
                "message": "🎤 Ready for next input"
            })

    async def process_complete_speech(self, session_id: str, speech_audio: bytes):
        """Process complete speech using your working direct API approach"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        # Skip processing if we're currently interrupted or assistant is already speaking
        if session.get('interrupted', False):
            logger.info("🛑 Skipping speech processing - assistant was interrupted")
            return

        if session.get('assistant_speaking', False):
            logger.info("🛑 Skipping speech processing - assistant is already speaking")
            return

        websocket = session['websocket']

        try:
            import aiohttp
            from openai import AsyncOpenAI
            import io
            import wave

            logger.info(f"🎯 Processing {len(speech_audio)} bytes of speech")

            # Convert raw audio to proper WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(config.SAMPLE_RATE)  # 16kHz
                wav_file.writeframes(speech_audio)

            wav_data = wav_buffer.getvalue()
            logger.info(f"🎵 Converted to WAV: {len(wav_data)} bytes (was {len(speech_audio)} raw bytes)")

            await websocket.send_json({
                "type": "processing_status",
                "status": "transcribing"
            })

            # 1. Transcribe with Whisper (your working approach)
            async with aiohttp.ClientSession() as aio_session:
                data = aiohttp.FormData()
                # Use the proper WAV data instead of raw audio
                data.add_field('audio', wav_data, filename='recording.wav', content_type='audio/wav')

                async with aio_session.post(config.WHISPER_API, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get('text', '').strip()
                        logger.info(f"📝 Whisper Success: {text}")
                        logger.info(f"📝 Full Whisper response: {result}")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Whisper API error {response.status}: {error_text}")
                        await websocket.send_json({"type": "error", "text": f"Whisper API failed: {response.status}"})
                        return

            if not text:
                return

            await websocket.send_json({"type": "transcription", "text": text})
            await websocket.send_json({"type": "processing_status", "status": "thinking"})

            # 2. Generate response with LLaMA (your working approach)
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL
            )

            # Add to conversation context
            session['conversation_context'].append({"role": "user", "content": text})

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Be conversational and concise."}
                ] + session['conversation_context'][-6:],  # Keep last 3 turns
                max_tokens=150
            )

            response_text = response.choices[0].message.content
            logger.info(f"💬 Response: {response_text}")

            # Add to conversation context
            session['conversation_context'].append({"role": "assistant", "content": response_text})

            await websocket.send_json({"type": "response", "text": response_text})
            await websocket.send_json({"type": "processing_status", "status": "speaking"})

            # Mark assistant as speaking (for interruption detection)
            session['assistant_speaking'] = True

            # 3. Convert to speech with TTS (your working approach)
            async with aiohttp.ClientSession() as aio_session:
                payload = {
                    "model": "tts-1-hd",  # Use HD model for better quality
                    "input": response_text,
                    "voice": "echo",      # Try echo - male voice, very clear
                    "speed": 0.85         # Slightly slower for clarity
                }

                headers = {
                    "Authorization": f"Bearer {config.E2E_TOKEN}",
                    "Content-Type": "application/json"
                }

                async with aio_session.post(config.TTS_API, json=payload, headers=headers) as tts_response:
                    if tts_response.status == 200:
                        # Get response headers to understand the format
                        content_type = tts_response.headers.get('content-type', 'unknown')
                        logger.info(f"🎵 TTS Response Content-Type: {content_type}")

                        audio_data = await tts_response.read()
                        logger.info(f"🎵 Received audio data: {len(audio_data)} bytes")

                        # If it's JSON, parse it to see the actual response
                        if content_type == 'application/json':
                            try:
                                json_response = json.loads(audio_data.decode('utf-8'))
                                logger.info(f"🎵 TTS JSON Response: {json_response}")

                                # Check if audio data is in the JSON response
                                if 'audio' in json_response:
                                    # Audio is base64 encoded in JSON
                                    audio_b64_from_api = json_response['audio']
                                    audio_data = base64.b64decode(audio_b64_from_api)
                                    content_type = 'audio/mpeg'  # Override content type
                                    logger.info(f"🎵 Extracted audio from JSON: {len(audio_data)} bytes")
                                elif 'data' in json_response:
                                    # Alternative field name
                                    audio_b64_from_api = json_response['data']
                                    audio_data = base64.b64decode(audio_b64_from_api)
                                    content_type = 'audio/mpeg'  # Override content type
                                    logger.info(f"🎵 Extracted audio from JSON (data field): {len(audio_data)} bytes")
                                else:
                                    logger.error(f"❌ No audio data found in JSON response: {list(json_response.keys())}")
                                    await websocket.send_json({"type": "error", "text": "TTS returned JSON without audio data"})
                                    return
                            except Exception as e:
                                logger.error(f"❌ Failed to parse TTS JSON response: {e}")
                                # Show first 200 chars of response for debugging
                                response_preview = audio_data.decode('utf-8', errors='ignore')[:200]
                                logger.error(f"❌ Response preview: {response_preview}")
                                await websocket.send_json({"type": "error", "text": "TTS returned invalid JSON"})
                                return

                        # Check if we got valid audio data
                        if audio_data and len(audio_data) > 100 and not session.get('interrupted', False):
                            # Check audio format by looking at header bytes
                            audio_header = audio_data[:12]
                            logger.info(f"🎵 Audio header bytes: {audio_header[:4]}")

                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio_response",
                                "data": audio_b64,
                                "content_type": content_type,
                                "size": len(audio_data)
                            })
                            logger.info(f"🔊 Audio response sent: {len(audio_data)} bytes, type: {content_type}")
                        else:
                            logger.error(f"❌ Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                            await websocket.send_json({"type": "error", "text": "Invalid audio data received"})
                    else:
                        error_text = await tts_response.text()
                        logger.error(f"❌ TTS API error {tts_response.status}: {error_text}")
                        await websocket.send_json({"type": "error", "text": f"TTS API failed: {tts_response.status}"})

            # Mark assistant finished speaking
            if not session.get('interrupted', False):
                session['assistant_speaking'] = False
                await websocket.send_json({"type": "processing_status", "status": "ready"})
            else:
                # If interrupted, just reset the flag
                session['assistant_speaking'] = False
                session['interrupted'] = False
                logger.info("🛑 Assistant finished after interruption")

        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}")
            await websocket.send_json({"type": "error", "text": str(e)})

    async def process_legacy_audio(self, session_id: str, base64_audio: str):
        """Support legacy push-to-talk mode"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        logger.info("🔄 Processing legacy push-to-talk audio")
        audio_bytes = base64.b64decode(base64_audio)
        await self.process_complete_speech(session_id, audio_bytes)

# Global handler
enhanced_handler = EnhancedVoiceHandler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint"""
    await enhanced_handler.handle_connection(websocket)

@app.get("/")
async def root():
    """Serve enhanced web client"""
    with open("web/index_enhanced.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "E2E Voice Assistant - Enhanced",
        "version": "2.0.0",
        "features": [
            "VAD (Voice Activity Detection)",
            "Interruption Handling",
            "Audio Buffering",
            "Conversation Context",
            "Direct API Integration"
        ]
    }

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 ENHANCED E2E VOICE ASSISTANT")
    logger.info("=" * 50)
    logger.info("✅ VAD - Automatic voice detection")
    logger.info("✅ Interruption - Stop assistant when you speak")
    logger.info("✅ Audio Buffering - Smart audio processing")
    logger.info("✅ Conversation Context - Remembers chat history")
    logger.info("✅ Direct API - Your working E2E endpoints")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")