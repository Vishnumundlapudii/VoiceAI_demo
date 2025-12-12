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
        """Enhanced VAD with adaptive thresholds and noise filtering"""
        try:
            import numpy as np
            import scipy.signal

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_frame.audio, dtype=np.int16)

            if len(audio_np) == 0:
                return False

            # Convert to float and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Apply high-pass filter to remove low-frequency noise
            nyquist = config.SAMPLE_RATE / 2
            cutoff = 100  # Hz - remove low frequency rumble
            b, a = scipy.signal.butter(2, cutoff / nyquist, btype='high')
            filtered_audio = scipy.signal.filtfilt(b, a, audio_float)

            # Multiple detection methods for robustness

            # 1. Energy-based detection
            energy = np.sqrt(np.mean(filtered_audio ** 2))

            # 2. Zero-crossing rate (helps distinguish speech from noise)
            zcr = np.mean(np.diff(np.sign(filtered_audio)) != 0)

            # 3. Spectral features
            fft = np.fft.fft(filtered_audio)
            magnitude = np.abs(fft[:len(fft)//2])

            # Focus on speech frequency range (300-3400 Hz)
            freqs = np.fft.fftfreq(len(filtered_audio), 1/config.SAMPLE_RATE)[:len(fft)//2]
            speech_band = (freqs >= 300) & (freqs <= 3400)
            speech_energy = np.sum(magnitude[speech_band])
            total_energy = np.sum(magnitude)

            speech_ratio = speech_energy / (total_energy + 1e-10)

            # Adaptive thresholds
            if assistant_speaking:
                # More sensitive during interruption
                energy_threshold = 0.008
                zcr_min, zcr_max = 0.05, 0.6
                speech_ratio_threshold = 0.3
                logger.debug(f"🎤 Interruption - Energy: {energy:.4f}, ZCR: {zcr:.3f}, SpeechRatio: {speech_ratio:.3f}")
            else:
                # Normal speech detection
                energy_threshold = 0.015
                zcr_min, zcr_max = 0.08, 0.5
                speech_ratio_threshold = 0.4
                logger.debug(f"🎤 Normal VAD - Energy: {energy:.4f}, ZCR: {zcr:.3f}, SpeechRatio: {speech_ratio:.3f}")

            # Combine multiple indicators
            energy_speech = energy > energy_threshold
            zcr_speech = zcr_min < zcr < zcr_max  # Speech has moderate ZCR
            spectral_speech = speech_ratio > speech_ratio_threshold

            # Decision logic: at least 2 out of 3 indicators must agree
            speech_indicators = [energy_speech, zcr_speech, spectral_speech]
            speech_detected = sum(speech_indicators) >= 2

            if speech_detected:
                confidence = sum(speech_indicators) / 3.0
                logger.debug(f"✅ Speech detected - Confidence: {confidence:.2f}")

                if assistant_speaking:
                    logger.info(f"🛑 INTERRUPTION DETECTED! Confidence: {confidence:.2f}")

            return speech_detected

        except Exception as e:
            logger.error(f"❌ Enhanced VAD detection error: {e}")
            # Fallback to simple energy detection
            try:
                audio_np = np.frombuffer(audio_frame.audio, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                threshold = 800 if assistant_speaking else 1200
                return volume > threshold
            except:
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
            current_time = time.time()
            session['last_speech_time'] = current_time
            session['speech_start_time'] = current_time  # Track when speech began
            session['interrupted'] = False  # Reset interruption flag

            # Enhanced interruption handling
            if session['assistant_speaking']:
                logger.info("🛑 FAST INTERRUPTION: User interrupted assistant")
                session['assistant_speaking'] = False
                session['interrupted'] = True
                session['interruption_time'] = time.time()

                # Send immediate stop signal with priority
                await websocket.send_json({
                    "type": "stop_audio",
                    "priority": "immediate",
                    "timestamp": time.time(),
                    "message": "Audio stopped immediately"
                })

                # Send interruption confirmation
                await websocket.send_json({
                    "type": "interruption",
                    "response_time_ms": 0,  # Immediate
                    "message": "⚡ Quick interruption - I'm listening"
                })

                # Set status to ready for user input
                await websocket.send_json({
                    "type": "processing_status",
                    "status": "ready",
                    "message": "Ready for your input"
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
        """Handle when no speech is detected with improved timing"""
        session = self.active_sessions.get(session_id)
        if not session or not session['is_speaking']:
            return

        # Check if enough silence has passed
        current_time = time.time()
        silence_duration = current_time - session['last_speech_time']

        # Get speech start time for total duration check
        speech_start_time = session.get('speech_start_time', current_time)
        total_speech_duration = current_time - speech_start_time

        # Use adaptive threshold based on speech length
        # Longer speeches need longer pauses to confirm end
        if total_speech_duration > 3.0:
            # For longer speeches, require longer silence
            required_silence = config.END_OF_SPEECH_THRESHOLD * 1.5
        else:
            required_silence = config.END_OF_SPEECH_THRESHOLD

        # Also check for absolute timeout to prevent infinite recording
        if silence_duration >= required_silence or total_speech_duration > config.SPEECH_TIMEOUT_THRESHOLD:
            # Speech ended - process the audio
            if total_speech_duration > config.SPEECH_TIMEOUT_THRESHOLD:
                logger.info(f"⏱️ VAD: Speech timeout after {total_speech_duration:.1f}s, processing...")
            else:
                logger.info(f"✅ VAD: Speech ended after {silence_duration:.1f}s silence, processing...")

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

            # Enhanced system prompt for professional responses
            system_prompt = """You are an intelligent and professional voice assistant. Your responses should be:

1. CLEAR & CONCISE: Give direct, well-structured answers
2. PROFESSIONAL: Use proper language, avoid slang, be courteous
3. HELPFUL: Provide actionable information and follow-up suggestions when appropriate
4. CONVERSATIONAL: Sound natural for voice interaction, but maintain professionalism
5. CONTEXTUAL: Remember the conversation flow and build upon previous exchanges

Guidelines:
- Keep responses between 20-50 words for voice interaction
- Use complete sentences with proper grammar
- Acknowledge user questions directly before providing information
- End with engagement when appropriate (e.g., "Would you like me to elaborate on any part?")
- If uncertain, clearly state limitations rather than guessing"""

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + session['conversation_context'][-8:],  # Keep last 4 turns for better context
                max_tokens=200,  # Increased for more complete responses
                temperature=0.7,  # Balanced creativity
                top_p=0.9,       # Focus on high-quality tokens
                presence_penalty=0.1,  # Encourage new topics
                frequency_penalty=0.1   # Reduce repetition
            )

            response_text = response.choices[0].message.content
            logger.info(f"💬 Response: {response_text}")

            # Add to conversation context
            session['conversation_context'].append({"role": "assistant", "content": response_text})

            await websocket.send_json({"type": "response", "text": response_text})
            await websocket.send_json({"type": "processing_status", "status": "speaking"})

            # Mark assistant as speaking (for interruption detection)
            session['assistant_speaking'] = True

            # 3. Convert to speech with Glow-TTS (high quality local server)
            async with aiohttp.ClientSession() as aio_session:
                # Use Glow-TTS server for better quality
                glow_tts_url = "http://216.48.191.105:8000/v1/audio/speech"
                payload = {
                    "input": response_text
                }

                headers = {
                    "Content-Type": "application/json"
                }

                logger.info(f"🎵 Calling Glow-TTS server: {glow_tts_url}")
                logger.info(f"🎵 TTS Input: {response_text[:50]}...")

                # Set shorter timeout and better error handling
                timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
                async with aio_session.post(glow_tts_url, json=payload, headers=headers, timeout=timeout) as tts_response:
                    logger.info(f"🎵 Glow-TTS response status: {tts_response.status}")
                    if tts_response.status == 200:
                        try:
                            # Glow-TTS returns JSON with base64 encoded audio
                            json_response = await tts_response.json()
                            logger.info(f"🎵 Glow-TTS Response received")
                            logger.info(f"🎵 Response keys: {list(json_response.keys())}")
                            logger.info(f"🎵 Response preview: {str(json_response)[:200]}")

                            # Extract audio from JSON response
                            if 'audio' in json_response:
                                audio_b64_from_api = json_response['audio']
                                logger.info(f"🎵 Extracted base64 audio from Glow-TTS: {len(audio_b64_from_api)} chars")

                                # Decode base64 to get WAV audio data
                                audio_data = base64.b64decode(audio_b64_from_api)
                                content_type = 'audio/wav'  # Glow-TTS returns WAV format
                                logger.info(f"🎵 Decoded WAV audio: {len(audio_data)} bytes")

                                # Enhanced interruption checking with timing
                                interruption_occurred = session.get('interrupted', False)
                                interruption_time = session.get('interruption_time', 0)
                                current_time = time.time()

                                # Check if interruption happened recently or is active
                                if audio_data and len(audio_data) > 100 and not interruption_occurred:
                                    # Double-check for interruption right before sending
                                    if not session.get('interrupted', False):
                                        # Send audio with interruption monitoring
                                        await websocket.send_json({
                                            "type": "audio_response",
                                            "data": audio_b64_from_api,
                                            "content_type": content_type,
                                            "size": len(audio_data),
                                            "format": "wav_22050_16bit_mono",
                                            "interruptible": True,  # Mark as interruptible
                                            "chunk_id": int(current_time * 1000)  # Unique chunk ID
                                        })
                                        logger.info(f"🔊 Interruptible audio sent: {len(audio_data)} bytes")
                                    else:
                                        logger.info("🛑 Audio blocked - interruption detected during send")
                                else:
                                    if interruption_occurred:
                                        interruption_delay = current_time - interruption_time
                                        logger.info(f"🛑 Audio skipped - interrupted {interruption_delay:.3f}s ago")
                                    else:
                                        logger.error(f"❌ Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                                        await websocket.send_json({"type": "error", "text": "Invalid audio data from Glow-TTS"})
                            else:
                                logger.error(f"❌ No 'audio' field in Glow-TTS response: {list(json_response.keys())}")
                                await websocket.send_json({"type": "error", "text": "Glow-TTS returned response without audio field"})

                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse Glow-TTS JSON response: {e}")
                            response_text = await tts_response.text()
                            logger.error(f"❌ Response preview: {response_text[:200]}")
                            await websocket.send_json({"type": "error", "text": "Glow-TTS returned invalid JSON"})
                        except Exception as e:
                            logger.error(f"❌ Glow-TTS processing error: {e}")
                            await websocket.send_json({"type": "error", "text": f"Glow-TTS error: {str(e)}"})
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

        except asyncio.TimeoutError:
            logger.error("❌ Glow-TTS request timeout - server took too long to respond")
            session['assistant_speaking'] = False
            await websocket.send_json({"type": "error", "text": "TTS timeout - please try again"})
        except aiohttp.ClientConnectorError as e:
            logger.error(f"❌ Cannot connect to Glow-TTS server: {e}")
            session['assistant_speaking'] = False
            await websocket.send_json({"type": "error", "text": "TTS server unavailable"})
        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}")
            session['assistant_speaking'] = False
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