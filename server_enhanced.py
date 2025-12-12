import base64
import io
import logging
import wave
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------------------------------------------------------------------
# Config import from clean_config.py (using your existing .env)
# -------------------------------------------------------------------
from config_clean import (
    WHISPER_API,
    TTS_API,
    LLAMA_BASE_URL,
    E2E_TOKEN,
    LLAMA_MODEL
)
# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice-backend")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Voice AI Backend with VAD",
    description="Whisper + LLaMA + TTS with simple VAD and 'no speech' handling.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Simple VAD (energy-based, tuned for 16 kHz audio)
# -------------------------------------------------------------------


def wav_bytes_to_float_mono(audio_bytes: bytes, expected_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Decode WAV bytes to a mono float32 numpy array in [-1, 1].
    Assumes 16-bit PCM. If anything is off, returns None.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            fr = wf.getframerate()
            n_frames = wf.getnframes()

            if sample_width != 2:
                logger.warning(f"Unexpected sample width: {sample_width * 8} bits")
            if fr != expected_rate:
                logger.warning(f"Unexpected sample rate: {fr}, expected {expected_rate}")
            if n_channels not in (1, 2):
                logger.warning(f"Unexpected channel count: {n_channels}")

            # Read raw PCM
            pcm_data = wf.readframes(n_frames)
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            # If stereo, convert to mono
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            return audio
    except Exception as e:
        logger.exception(f"Failed to decode WAV bytes: {e}")
        return None


def simple_vad(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    energy_threshold: float = 0.0005,
    min_voiced_ms: int = 300,
) -> bool:
    """
    Very simple energy-based VAD:
    - Split signal into frames (e.g., 30 ms)
    - Compute mean energy of each frame
    - Count frames above threshold
    - If total voiced duration >= min_voiced_ms -> speech present
    """
    if audio is None or len(audio) == 0:
        return False

    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size <= 0:
        return False

    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return False

    audio = audio[: n_frames * frame_size]
    frames = audio.reshape(n_frames, frame_size)
    energies = np.mean(frames * frames, axis=1)

    voiced_frames = energies > energy_threshold
    voiced_ms = voiced_frames.sum() * frame_ms

    logger.info(
        f"VAD: total_frames={n_frames}, voiced_frames={voiced_frames.sum()}, "
        f"voiced_ms={voiced_ms}, thresh={energy_threshold}"
    )

    return voiced_ms >= min_voiced_ms


# -------------------------------------------------------------------
# API helpers (Whisper, LLaMA, TTS)
# -------------------------------------------------------------------


def call_whisper(audio_bytes: bytes) -> Optional[str]:
    """
    Send raw audio bytes to the Whisper API and return transcript text.
    Adjust the payload to whatever your Whisper endpoint expects.
    """
    try:
        logger.info("Sending audio to Whisper API...")
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }
        resp = requests.post(WHISPER_API, files=files, timeout=60)

        if resp.status_code != 200:
            logger.error(f"Whisper API error: {resp.status_code} - {resp.text}")
            return None

        data = resp.json()
        if "text" in data:
            text = data["text"]
        elif "result" in data and isinstance(data["result"], dict) and "text" in data["result"]:
            text = data["result"]["text"]
        else:
            logger.error(f"Unexpected Whisper response format: {data}")
            return None

        text = (text or "").strip()
        logger.info(f"Whisper transcript: {text}")
        return text if text else None

    except Exception as e:
        logger.exception(f"Whisper call failed: {e}")
        return None


def call_llama(user_text: str) -> Optional[str]:
    """
    Call LLaMA using your existing infer endpoint.
    Uses OpenAI-compatible /chat/completions shape.
    """
    try:
        logger.info("Calling LLaMA for response...")

        if LLAMA_BASE_URL.endswith("/chat/completions"):
            url = LLAMA_BASE_URL
        else:
            url = f"{LLAMA_BASE_URL.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {E2E_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, concise voice assistant. "
                        "Stay on the current topic and give short clear answers."
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            "temperature": 0.4,
            "max_tokens": 256,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        if resp.status_code != 200:
            logger.error(f"LLaMA API error: {resp.status_code} - {resp.text}")
            return None

        data = resp.json()
        try:
            reply = data["choices"][0]["message"]["content"].strip()
            logger.info(f"LLaMA reply: {reply}")
            return reply
        except Exception:
            logger.error(f"Unexpected LLaMA response format: {data}")
            return None

    except Exception as e:
        logger.exception(f"LLaMA call failed: {e}")
        return None


def call_tts(text: str) -> Optional[bytes]:
    """
    Call your existing TTS API:
      POST TTS_API
      Body: {"input": "<text>"}
      Response: raw MP3 bytes
    """
    try:
        logger.info("Calling TTS API...")
        payload = {"input": text}
        resp = requests.post(TTS_API, json=payload, timeout=60)

        if resp.status_code != 200:
            logger.error(f"TTS API error: {resp.status_code} - {resp.text}")
            return None

        audio_bytes = resp.content
        logger.info(f"TTS audio size: {len(audio_bytes)} bytes")
        return audio_bytes

    except Exception as e:
        logger.exception(f"TTS call failed: {e}")
        return None


# -------------------------------------------------------------------
# Main endpoint with VAD + "no speech" guard
# -------------------------------------------------------------------


@app.post("/v1/voice/query")
async def voice_query(audio: UploadFile = File(...)):
    """
    Full pipeline with VAD:
      1. Receive audio from UI
      2. Decode WAV -> run VAD
         - If no speech -> return {"status": "no_speech"} (UI can show [No speech detected])
      3. Send audio to Whisper -> transcript
         - If transcript is empty/too short -> also treat as no speech
      4. Call LLaMA -> reply_text
      5. Call TTS -> audio bytes
      6. Return JSON:
         {
           "status": "ok",
           "transcript": "...",
           "reply_text": "...",
           "audio_base64": "<mp3>"
         }
    """
    try:
        audio_bytes = await audio.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            logger.warning("Received too little audio data.")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_speech",
                    "message": "No speech detected in audio (too little data).",
                },
            )

        # --- Step 1: local VAD before hitting Whisper ---
        float_audio = wav_bytes_to_float_mono(audio_bytes, expected_rate=16000)
        if float_audio is None:
            logger.warning("Could not decode audio for VAD; falling back to Whisper only.")
        else:
            has_speech = simple_vad(float_audio)
            if not has_speech:
                logger.info("VAD says: no speech, skipping Whisper/LLM/TTS.")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "no_speech",
                        "message": "No speech detected by VAD.",
                    },
                )

        # --- Step 2: Whisper STT ---
        transcript = call_whisper(audio_bytes)
        if not transcript:
            logger.info("Whisper returned empty transcript; treating as no speech.")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_speech",
                    "message": "Could not transcribe any speech.",
                },
            )

        # Very short 'hmm', 'uh', etc. -> also skip
        if len(transcript.split()) < 2 and len(transcript) < 6:
            logger.info(f"Transcript too short ('{transcript}'); treating as no speech.")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_speech",
                    "message": "Transcript too short, likely no real speech.",
                    "transcript": transcript,
                },
            )

        # --- Step 3: LLaMA reply ---
        reply_text = call_llama(transcript)
        if not reply_text:
            raise HTTPException(status_code=502, detail="LLM error while generating response.")

        # --- Step 4: TTS ---
        tts_audio = call_tts(reply_text)
        if not tts_audio:
            raise HTTPException(status_code=502, detail="TTS error while generating speech.")

        audio_b64 = base64.b64encode(tts_audio).decode("utf-8")

        return {
            "status": "ok",
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_base64": audio_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /v1/voice/query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# -------------------------------------------------------------------
# Health check (optional, but nice for debugging)
# -------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------------------------
# Uvicorn entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_enhanced:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
