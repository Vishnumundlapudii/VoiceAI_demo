"""
Fast WebSocket Voice Demo with VAD and Interruption
Created for demo - optimized for reliability and speed
"""

import asyncio
import json
import base64
import time
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import aiohttp
from openai import AsyncOpenAI
import io
import wave
import numpy as np

from loguru import logger
import config_clean as config

app = FastAPI(title="Demo Voice Assistant - WebSocket + VAD + Interruption")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DemoVoiceHandler:
    """Fast demo handler with essential features only"""

    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        logger.info("🎤 Demo Voice Handler - WebSocket + VAD + Interruption Ready!")

    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        session_id = id(websocket)

        try:
            await websocket.accept()
            logger.info(f"🔗 Demo session connected: {session_id}")

            # Initialize session
            self.active_sessions[session_id] = {
                "websocket": websocket,
                "audio_buffer": bytearray(),
                "is_speaking": False,
                "assistant_speaking": False,
                "last_speech_time": None,
                "processing": False,
            }

            await websocket.send_json({
                "type": "connection_status",
                "status": "connected",
                "message": "🎤 Demo ready! VAD + Interruption active"
            })

            # Main message loop
            while True:
                data = await websocket.receive_json()

                if data.get("type") == "audio_chunk":
                    await self.process_audio_chunk(session_id, data["data"])

        except WebSocketDisconnect:
            logger.info(f"❌ Demo session disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Demo session error: {e}")
        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def process_audio_chunk(self, session_id: str, base64_audio: str):
        """Process audio with fast VAD"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        try:
            audio_bytes = base64.b64decode(base64_audio)
            websocket = session["websocket"]

            # Fast VAD using RMS
            speech_detected = self.detect_speech_fast(audio_bytes)

            if speech_detected:
                await self.handle_speech_detected(session_id, audio_bytes)
            else:
                if session["is_speaking"]:
                    await self.handle_silence(session_id)

        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")

    def detect_speech_fast(self, audio_bytes: bytes) -> bool:
        """Ultra-fast RMS-based speech detection"""
        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_np) == 0:
                return False

            # RMS volume calculation
            rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))

            # Use threshold from config
            threshold = getattr(config, "VAD_ENERGY_THRESHOLD", 800.0)

            speech_detected = rms > threshold

            if speech_detected:
                logger.debug(f"🗣️ Speech: RMS={rms:.1f} > {threshold}")

            return speech_detected

        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return False

    async def handle_speech_detected(self, session_id: str, audio_bytes: bytes):
        """Handle when speech starts/continues"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session["websocket"]
        current_time = time.time()

        if not session["is_speaking"]:
            # Speech just started
            logger.info("🗣️ Speech started")
            session["is_speaking"] = True
            session["audio_buffer"] = bytearray()
            session["speech_start_time"] = current_time

            # INTERRUPTION DETECTION
            if session["assistant_speaking"]:
                logger.info("🛑 INTERRUPTION! Stopping assistant")
                session["assistant_speaking"] = False

                # Send stop signal
                await websocket.send_json({
                    "type": "stop_audio",
                    "message": "⚡ Interrupted - I'm listening"
                })

            await websocket.send_json({
                "type": "vad_status",
                "status": "speaking",
                "message": "👂 Listening..."
            })

        # Add audio to buffer
        session["audio_buffer"].extend(audio_bytes)
        session["last_speech_time"] = current_time

    async def handle_silence(self, session_id: str):
        """Handle when speech ends (silence detected)"""
        session = self.active_sessions.get(session_id)
        if not session or not session["is_speaking"]:
            return

        current_time = time.time()
        silence_duration = current_time - session["last_speech_time"]

        # Wait for enough silence before processing
        if silence_duration >= config.END_OF_SPEECH_THRESHOLD:
            logger.info(f"✅ Speech ended after {silence_duration:.1f}s silence")
            session["is_speaking"] = False

            if len(session["audio_buffer"]) > 0:
                await self.process_complete_speech(session_id, bytes(session["audio_buffer"]))
                session["audio_buffer"] = bytearray()

            await session["websocket"].send_json({
                "type": "vad_status",
                "status": "ready",
                "message": "🎤 Ready for input"
            })

    async def process_complete_speech(self, session_id: str, speech_audio: bytes):
        """Process complete speech - simplified for demo reliability"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        websocket = session["websocket"]

        # Prevent concurrent processing
        if session["processing"]:
            logger.info("⏭️ Already processing, skipping")
            return

        session["processing"] = True

        try:
            # Quick audio validation - be less aggressive about short audio
            audio_duration = len(speech_audio) / (config.SAMPLE_RATE * 2)
            if audio_duration < 0.05:  # Very short threshold
                logger.warning("⚠️ Audio extremely short, skipping")
                return

            logger.info(f"🎯 Processing {audio_duration:.2f}s of speech")

            # Convert to WAV for Whisper
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(config.SAMPLE_RATE)
                wav_file.writeframes(speech_audio)

            wav_data = wav_buffer.getvalue()

            await websocket.send_json({"type": "processing_status", "status": "transcribing"})

            # 1. Whisper transcription
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session_http:
                data = aiohttp.FormData()
                data.add_field("audio", wav_data, filename="audio.wav", content_type="audio/wav")

                async with session_http.post(config.WHISPER_API, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("text", "").strip()
                        logger.info(f"📝 Whisper: '{text}'")

                        if not text:
                            logger.warning("⚠️ Empty transcription")
                            return
                    else:
                        logger.error(f"❌ Whisper error: {response.status}")
                        await websocket.send_json({"type": "error", "text": "Transcription failed"})
                        return

            await websocket.send_json({"type": "transcription", "text": text})
            await websocket.send_json({"type": "processing_status", "status": "thinking"})

            # 2. LLaMA response
            client = AsyncOpenAI(api_key=config.E2E_TOKEN, base_url=config.LLAMA_BASE_URL)

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and clear."},
                    {"role": "user", "content": text}
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )

            response_text = response.choices[0].message.content
            logger.info(f"💬 LLaMA: {response_text}")

            await websocket.send_json({"type": "response", "text": response_text})
            await websocket.send_json({"type": "processing_status", "status": "speaking"})

            # Mark assistant as speaking (for interruption detection)
            session["assistant_speaking"] = True

            # 3. TTS
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session_http:
                payload = {"input": response_text}
                headers = {"Content-Type": "application/json"}

                async with session_http.post(config.TTS_API, json=payload, headers=headers) as tts_response:
                    if tts_response.status == 200:
                        json_response = await tts_response.json()

                        if "audio" in json_response:
                            audio_b64 = json_response["audio"]

                            # Check if still should play (not interrupted)
                            if session.get("assistant_speaking", False):
                                await websocket.send_json({
                                    "type": "audio_response",
                                    "data": audio_b64,
                                    "interruptible": True
                                })
                                logger.info("🔊 Audio sent")
                            else:
                                logger.info("🛑 Audio skipped - interrupted")
                        else:
                            logger.error("❌ No audio in TTS response")
                    else:
                        logger.error(f"❌ TTS error: {tts_response.status}")

            # Mark assistant finished speaking
            if not session.get("interrupted", False):
                session["assistant_speaking"] = False
                await websocket.send_json({"type": "processing_status", "status": "ready"})

        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            await websocket.send_json({"type": "error", "text": str(e)})
        finally:
            session["processing"] = False

# Global handler
demo_handler = DemoVoiceHandler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Demo WebSocket endpoint"""
    await demo_handler.handle_connection(websocket)

@app.get("/")
async def root():
    """Serve web client"""
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Demo Voice Assistant", "features": ["WebSocket", "VAD", "Interruption"]}

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 DEMO VOICE ASSISTANT - WebSocket + VAD + Interruption")
    logger.info("=" * 50)
    logger.info("✅ Fast VAD - RMS-based speech detection")
    logger.info("✅ Real-time interruption - Stop assistant anytime")
    logger.info("✅ WebSocket streaming - Natural conversation flow")
    logger.info(f"🎤 VAD Threshold: {config.VAD_ENERGY_THRESHOLD}")
    logger.info("🎯 Optimized for demo reliability!")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")