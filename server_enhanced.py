import asyncio
import base64
import io
import logging
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
from TTS.api import TTS

# ---------------------------------------------------------------------
# Basic config
# ---------------------------------------------------------------------

SAMPLE_RATE = 16000          # 🔴 Change this if your frontend uses a different rate
MIN_TRANSCRIPT_CHARS = 3     # below this, ignore as noise/silence
SILENCE_ENERGY_THRESHOLD = 500  # tweak if needed
MAX_SILENCE_SECONDS = 15     # optional: reset conversation after long silence

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("voice-backend")

app = FastAPI(title="Voice Assistant Backend")

# CORS – update allowed origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# TTS Initialization (Glow-TTS)
# ---------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Loading Glow-TTS on {device}")
tts = TTS("tts_models/en/ljspeech/glow-tts").to(device)
logger.info("✅ Glow-TTS ready!")

# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str

class TTSSpeechResponse(BaseModel):
    audio_base64: str
    format: str = "mp3"

class ChatTextRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = None  # [{"role": "user"/"assistant", "content": "..."}, ...]

class ChatResponse(BaseModel):
    reply: str

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def is_silent_pcm16(audio_bytes: bytes, sample_rate: int) -> bool:
    """
    Very simple silence detector based on average absolute amplitude.
    """
    if not audio_bytes:
        return True

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_np.size == 0:
        return True

    energy = np.mean(np.abs(audio_np))
    logger.debug(f"🔍 Audio energy: {energy:.2f}")

    return energy < SILENCE_ENERGY_THRESHOLD


def generate_speech_mp3(text: str) -> bytes:
    """
    Use Glow-TTS to generate waveform and encode as MP3.
    """
    logger.info(f"🗣️ Generating TTS for text: {text[:80]!r}")
    # TTS returns a 1D float32 numpy array at 22050 Hz by default
    wav = tts.tts(text)
    wav = np.asarray(wav, dtype=np.float32)

    # Normalise to int16 for pydub
    wav_int16 = np.int16(wav / np.max(np.abs(wav)) * 32767)

    # Convert to AudioSegment (assume 22050Hz mono)
    audio_seg = AudioSegment(
        wav_int16.tobytes(),
        frame_rate=22050,
        sample_width=2,
        channels=1,
    )

    # Export as MP3 to bytes buffer
    buf = io.BytesIO()
    audio_seg.export(buf, format="mp3")
    buf.seek(0)
    mp3_bytes = buf.read()
    logger.info(f"✅ TTS MP3 generated, {len(mp3_bytes)} bytes")
    return mp3_bytes


async def dummy_llm_reply(user_text: str, history: Optional[List[dict]] = None) -> str:
    """
    Placeholder LLM. Replace this with your actual model call.
    This function is where hallucinations could start if called on junk text.
    """
    # Here you would call OpenAI / local LLM etc.
    # Make sure to pass history if you want conversation context.
    await asyncio.sleep(0.05)
    return f"You said: {user_text}"


# ---------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/audio/speech", response_model=TTSSpeechResponse)
async def tts_endpoint(req: TTSRequest):
    """
    Text → Glow-TTS → MP3 (base64).
    """
    text = (req.text or "").strip()
    if not text:
        return TTSSpeechResponse(audio_base64="", format="mp3")

    mp3_bytes = generate_speech_mp3(text)
    b64 = base64.b64encode(mp3_bytes).decode("utf-8")
    return TTSSpeechResponse(audio_base64=b64, format="mp3")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatTextRequest):
    """
    Simple text chat endpoint using dummy LLM.
    This shares the same guardrails as the voice flow.
    """
    text = (req.text or "").strip()

    # Guard: ignore empty / extremely short input
    if len(text) < MIN_TRANSCRIPT_CHARS:
        logger.info("⚠️ Ignoring very short text input to avoid nonsense LLM calls.")
        return ChatResponse(reply="")

    reply = await dummy_llm_reply(text, req.history)
    return ChatResponse(reply=reply)


# ---------------------------------------------------------------------
# WebSocket for voice conversation (template)
# ---------------------------------------------------------------------
# Protocol assumption:
# - Client sends JSON messages:
#   { "type": "audio", "audio_base64": "<...>" }  → single PCM16 chunk
#   { "type": "end_utterance" }                  → finish current turn, run STT + LLM + TTS
#   { "type": "reset" }                          → clear history
#
# - Server sends:
#   { "type": "transcript", "text": "..." }
#   { "type": "reply_text", "text": "..." }
#   { "type": "reply_audio", "audio_base64": "<mp3>" }
#   { "type": "info", "message": "..." }
#
# You can adapt this to match your real frontend exactly.
# ---------------------------------------------------------------------

class ConversationState:
    def __init__(self):
        self.buffers: List[bytes] = []
        self.history: List[dict] = []
        self.last_speech_ts: float = asyncio.get_event_loop().time()

    def reset_audio(self):
        self.buffers.clear()

    def reset_all(self):
        self.buffers.clear()
        self.history.clear()

    def append_audio(self, chunk: bytes):
        self.buffers.append(chunk)
        self.last_speech_ts = asyncio.get_event_loop().time()

    def get_pcm(self) -> bytes:
        if not self.buffers:
            return b""
        return b"".join(self.buffers)


@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """
    Voice WebSocket with:
    - silence detection
    - transcript length guard
    - optional history reset on long silence
    - no random replies when no speech is detected
    """
    await ws.accept()
    logger.info("🌐 Voice WebSocket connected")

    state = ConversationState()

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            # 1) Reset conversation explicitly from UI
            if msg_type == "reset":
                logger.info("🔁 Conversation reset requested from UI")
                state.reset_all()
                await ws.send_json({"type": "info", "message": "conversation_reset"})
                continue

            # 2) Audio chunk
            if msg_type == "audio":
                audio_b64 = msg.get("audio_base64", "")
                if not audio_b64:
                    continue

                try:
                    chunk = base64.b64decode(audio_b64)
                except Exception as e:
                    logger.warning(f"Failed to base64-decode audio chunk: {e}")
                    continue

                # Optional: quick silence check per chunk if you want to show "No speech detected" live
                if is_silent_pcm16(chunk, SAMPLE_RATE):
                    # You may or may not want to notify the UI here.
                    await ws.send_json({"type": "info", "message": "no_speech_chunk"})
                else:
                    state.append_audio(chunk)

                continue

            # 3) End of utterance: run STT + LLM + TTS
            if msg_type == "end_utterance":
                full_pcm = state.get_pcm()
                state.reset_audio()

                # If no meaningful audio collected
                if not full_pcm or is_silent_pcm16(full_pcm, SAMPLE_RATE):
                    logger.info("⚠️ End utterance but no speech detected; skipping STT + LLM.")
                    await ws.send_json({"type": "info", "message": "no_speech_detected"})
                    # IMPORTANT: do NOT call LLM here → avoids random topic jumps
                    continue

                # -----------------------------------------------------------------
                # TODO: Replace this with your Whisper / ASR call
                # For now we simulate a transcript.
                # -----------------------------------------------------------------
                # Example if you use openai-whisper:
                # import whisper
                # model = whisper.load_model("small", device=device)
                # audio_float = np.frombuffer(full_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                # result = model.transcribe(audio_float, language="en")
                # transcript = (result.get("text") or "").strip()
                # -----------------------------------------------------------------
                transcript = "dummy transcript from audio"  # placeholder
                logger.info(f"📝 Transcript: {transcript!r}")
                await ws.send_json({"type": "transcript", "text": transcript})

                # Guard: if transcript is too short/empty, DO NOT call LLM
                if not transcript or len(transcript) < MIN_TRANSCRIPT_CHARS:
                    logger.info("⚠️ Transcript too short; skipping LLM/TTS to avoid hallucinations.")
                    await ws.send_json({"type": "info", "message": "transcript_too_short"})
                    continue

                # Build history for LLM
                state.history.append({"role": "user", "content": transcript})

                # Call LLM safely
                reply_text = await dummy_llm_reply(transcript, state.history)
                logger.info(f"🤖 LLM reply: {reply_text!r}")
                state.history.append({"role": "assistant", "content": reply_text})

                await ws.send_json({"type": "reply_text", "text": reply_text})

                # TTS for reply
                mp3_bytes = generate_speech_mp3(reply_text)
                reply_audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
                await ws.send_json({"type": "reply_audio", "audio_base64": reply_audio_b64})

                continue

            # 4) Unknown message type
            logger.warning(f"Unknown WebSocket message type: {msg_type!r}")
            await ws.send_json({"type": "info", "message": f"unknown_type:{msg_type}"})

            # 5) Optional: long silence → reset conversation
            now = asyncio.get_event_loop().time()
            if now - state.last_speech_ts > MAX_SILENCE_SECONDS and state.history:
                logger.info("⏱️ Long silence detected, resetting conversation history.")
                state.reset_all()
                await ws.send_json({"type": "info", "message": "auto_reset_due_to_silence"})

    except WebSocketDisconnect:
        logger.info("🔌 Voice WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": "internal_server_error"})
        except Exception:
            pass


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_enhanced:app",   # filename:app_name
        host="0.0.0.0",
        port=8000,
        reload=True,             # disable in production
    )
