"""
Simple, Reliable Voice Assistant Server
Clean implementation with simple VAD that actually works
"""

import asyncio
import json
import base64
import time
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import aiohttp
import uvicorn
import numpy as np
import io
import wave
from openai import AsyncOpenAI

from loguru import logger
import config

app = FastAPI(title="Simple Voice Assistant")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleVoiceHandler:
    """Simple, reliable voice handler with basic VAD"""

    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        logger.info("✅ Simple Voice Handler initialized")

    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        session_id = id(websocket)
        try:
            await websocket.accept()
            logger.info(f"✅ Connected: {session_id}")

            # Initialize session
            self.active_sessions[session_id] = {
                'websocket': websocket,
                'audio_buffer': bytearray(),
                'is_speaking': False,
                'last_speech_time': None,
                'conversation_context': []
            }

            await websocket.send_json({
                "type": "connection_status",
                "status": "connected",
                "message": "🎤 Simple VAD ready - speak clearly!"
            })

            # Main message loop
            while True:
                data = await websocket.receive_json()

                if data["type"] == "audio_chunk":
                    await self.process_audio_chunk(session_id, data["data"])

        except WebSocketDisconnect:
            logger.info(f"❌ Disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Error {session_id}: {e}")
        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def process_audio_chunk(self, session_id: str, base64_audio: str):
        """Process audio with simple VAD"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        try:
            # Decode audio
            audio_bytes = base64.b64decode(base64_audio)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_np) == 0:
                return

            # Simple energy-based speech detection
            energy = np.mean(audio_np.astype(np.float64) ** 2)

            # Clear threshold - no false positives
            speech_detected = energy > 200000000  # Adjust this value as needed

            if speech_detected:
                await self.handle_speech_detected(session_id, audio_bytes)
            else:
                if session['is_speaking']:
                    await self.handle_no_speech(session_id)

        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")

    async def handle_speech_detected(self, session_id: str, audio_bytes: bytes):
        """Handle speech detection"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session['websocket']

        if not session['is_speaking']:
            # Speech started
            logger.info("🗣️ Speech started")
            session['is_speaking'] = True
            session['audio_buffer'] = bytearray()
            session['speech_start_time'] = time.time()

        session['audio_buffer'].extend(audio_bytes)
        session['last_speech_time'] = time.time()

    async def handle_no_speech(self, session_id: str):
        """Handle when speech ends"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        current_time = time.time()
        silence_duration = current_time - session['last_speech_time']

        # Use config value - 1.5 seconds of silence ends speech
        if silence_duration >= config.END_OF_SPEECH_THRESHOLD:
            logger.info("✅ Speech ended - processing")
            session['is_speaking'] = False

            if len(session['audio_buffer']) > 0:
                await self.process_complete_speech(session_id, bytes(session['audio_buffer']))

    async def process_complete_speech(self, session_id: str, audio_data: bytes):
        """Process complete speech"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session['websocket']

        try:
            # Convert to WAV
            wav_data = self.convert_to_wav(audio_data)

            # 1. Transcribe with Whisper
            async with aiohttp.ClientSession() as aio_session:
                data = aiohttp.FormData()
                data.add_field('audio', wav_data, filename='recording.wav', content_type='audio/wav')

                async with aio_session.post(config.WHISPER_API, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get('text', '').strip()

                        if not text:
                            logger.warning("Empty transcription")
                            return

                        logger.info(f"📝 Transcribed: '{text}'")
                    else:
                        logger.error(f"Whisper error: {response.status}")
                        return

            # 2. Get LLM response
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL
            )

            session['conversation_context'].append({"role": "user", "content": text})

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses conversational and concise."}
                ] + session['conversation_context'][-10:],
                max_tokens=150,
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()
            session['conversation_context'].append({"role": "assistant", "content": response_text})

            await websocket.send_json({"type": "response", "text": response_text})
            logger.info(f"💬 Response: {response_text}")

            # 3. Convert to speech
            async with aiohttp.ClientSession() as aio_session:
                payload = {"input": response_text}
                headers = {"Content-Type": "application/json"}
                tts_url = "http://216.48.191.105:8000/v1/audio/speech"

                async with aio_session.post(tts_url, json=payload, headers=headers) as tts_response:
                    if tts_response.status == 200:
                        tts_result = await tts_response.json()
                        audio_b64 = tts_result.get('audio', '')

                        if audio_b64:
                            await websocket.send_json({
                                "type": "audio_response",
                                "data": audio_b64,
                                "content_type": "audio/wav"
                            })
                            logger.info("🔊 Audio sent")

        except Exception as e:
            logger.error(f"❌ Processing error: {e}")

    def convert_to_wav(self, audio_data: bytes) -> bytes:
        """Convert raw audio to WAV format"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(config.SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        return wav_buffer.getvalue()

# Initialize handler
handler = SimpleVoiceHandler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handler.handle_connection(websocket)

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Voice Assistant</title>
    </head>
    <body>
        <h1>Simple Voice Assistant</h1>
        <p>WebSocket server running on /ws</p>
        <p>Simple, reliable VAD - no complex workarounds</p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Voice Assistant Server")
    logger.info("✅ Simple VAD - just energy threshold")
    logger.info("✅ No complex processing")
    logger.info("✅ Reliable speech detection")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")