"""
CLEAN SIMPLE VOICE ASSISTANT SERVER
Uses unified configuration - no more scattered parameters!
"""

import asyncio
import json
import base64
import time
import aiohttp
import uvicorn
import numpy as np
import io
import wave
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from loguru import logger

# Import our clean, unified configuration
import config_clean as config

app = FastAPI(title="Clean Voice Assistant")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CleanVoiceHandler:
    """Simple, reliable voice handler with unified configuration"""

    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        logger.info("✅ Clean Voice Handler initialized with unified config")

    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        session_id = id(websocket)
        try:
            await websocket.accept()
            logger.info(f"✅ Connected: {session_id}")

            # Initialize session using config values
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
                "message": f"🎤 Clean VAD ready - energy threshold: {config.VAD_ENERGY_THRESHOLD:,}"
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
        """Process audio with clean VAD using unified config"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        try:
            # Decode audio
            audio_bytes = base64.b64decode(base64_audio)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_np) == 0:
                return

            # Simple energy-based speech detection using config value
            energy = np.mean(audio_np.astype(np.float64) ** 2)
            speech_detected = energy > config.VAD_ENERGY_THRESHOLD

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

        if not session['is_speaking']:
            # Speech started
            logger.info("🗣️ Speech started")
            session['is_speaking'] = True
            session['audio_buffer'] = bytearray()
            session['speech_start_time'] = time.time()

        session['audio_buffer'].extend(audio_bytes)
        session['last_speech_time'] = time.time()

    async def handle_no_speech(self, session_id: str):
        """Handle when speech ends using unified config"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        current_time = time.time()
        silence_duration = current_time - session['last_speech_time']

        # Use config value for speech end threshold
        if silence_duration >= config.END_OF_SPEECH_THRESHOLD:
            logger.info(f"✅ Speech ended after {silence_duration:.1f}s - processing")
            session['is_speaking'] = False

            if len(session['audio_buffer']) > 0:
                await self.process_complete_speech(session_id, bytes(session['audio_buffer']))

    async def process_complete_speech(self, session_id: str, audio_data: bytes):
        """Process complete speech using unified configuration"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session['websocket']

        try:
            # Check minimum audio duration using config
            audio_duration = len(audio_data) / (config.SAMPLE_RATE * 2)
            if audio_duration < config.MIN_AUDIO_DURATION:
                logger.warning(f"⚠️ Audio too short: {audio_duration:.3f}s, skipping")
                return

            # Convert to WAV using config values
            wav_data = self.convert_to_wav(audio_data)

            # 1. Transcribe with Whisper using config timeout
            timeout = aiohttp.ClientTimeout(total=config.WHISPER_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as aio_session:
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

            # 2. Get LLM response using unified config
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL,
                timeout=config.LLM_TIMEOUT
            )

            session['conversation_context'].append({"role": "user", "content": text})

            # Use all config values for LLM
            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT}
                ] + session['conversation_context'][-config.CONVERSATION_CONTEXT_LENGTH:],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                presence_penalty=config.LLM_PRESENCE_PENALTY,
                frequency_penalty=config.LLM_FREQUENCY_PENALTY
            )

            response_text = response.choices[0].message.content.strip()
            session['conversation_context'].append({"role": "assistant", "content": response_text})

            await websocket.send_json({"type": "response", "text": response_text})
            logger.info(f"💬 Response: {response_text}")

            # 3. Convert to speech using config timeout
            timeout = aiohttp.ClientTimeout(total=config.TTS_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as aio_session:
                payload = {"input": response_text}
                headers = {"Content-Type": "application/json"}

                async with aio_session.post(config.TTS_API, json=payload, headers=headers) as tts_response:
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
        """Convert raw audio to WAV format using config values"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(config.CHANNELS)
            wav_file.setsampwidth(config.AUDIO_BIT_DEPTH // 8)  # Convert bits to bytes
            wav_file.setframerate(config.SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        return wav_buffer.getvalue()

# Initialize handler
handler = CleanVoiceHandler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handler.handle_connection(websocket)

@app.get("/")
async def get():
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clean Voice Assistant</title>
    </head>
    <body>
        <h1>🧹 Clean Voice Assistant</h1>
        <p><strong>WebSocket:</strong> /ws</p>

        <h3>📊 Configuration:</h3>
        <ul>
            <li><strong>VAD Energy Threshold:</strong> {config.VAD_ENERGY_THRESHOLD:,}</li>
            <li><strong>Speech End Timeout:</strong> {config.END_OF_SPEECH_THRESHOLD}s</li>
            <li><strong>LLM Max Tokens:</strong> {config.LLM_MAX_TOKENS}</li>
            <li><strong>Context Length:</strong> {config.CONVERSATION_CONTEXT_LENGTH} messages</li>
            <li><strong>Audio:</strong> {config.SAMPLE_RATE}Hz, {config.CHANNELS} channel, {config.AUDIO_BIT_DEPTH}-bit</li>
        </ul>

        <p>✅ <strong>All parameters unified in config_clean.py</strong></p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info("🚀 Starting CLEAN Voice Assistant Server")
    logger.info("🧹 Using unified configuration - no more parameter chaos!")
    logger.info(f"🎤 VAD Energy Threshold: {config.VAD_ENERGY_THRESHOLD:,}")
    logger.info(f"⏱️ Speech Timeout: {config.END_OF_SPEECH_THRESHOLD}s")
    logger.info(f"🧠 LLM Config: {config.LLM_MAX_TOKENS} tokens, temp={config.LLM_TEMPERATURE}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")