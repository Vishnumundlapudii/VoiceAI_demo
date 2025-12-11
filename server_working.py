"""
WORKING WebSocket Server - Direct Integration
No complex transports, just direct API calls
"""

import asyncio
import json
import base64
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="E2E Voice Assistant - WORKING")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
WHISPER_API = os.getenv("WHISPER_API", "http://101.53.140.94:8000/transcribe")
TTS_API = os.getenv("TTS_API", "http://101.53.140.105:8000/v1/audio/speech")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "https://infer.e2enetworks.net/project/p-5861/endpoint/is-7619/v1")
E2E_TOKEN = os.getenv("E2E_TOKEN")


async def transcribe_audio(audio_data: bytes) -> str:
    """Call Whisper API"""
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')

            async with session.post(WHISPER_API, data=data) as response:
                result = await response.json()
                return result.get('text', '')
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return ""


async def generate_response(text: str) -> str:
    """Call LLaMA API"""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=E2E_TOKEN,
            base_url=LLAMA_BASE_URL
        )

        response = await client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
                {"role": "user", "content": text}
            ],
            max_tokens=100
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLaMA error: {e}")
        return "I'm sorry, I couldn't process that."


async def text_to_speech(text: str) -> bytes:
    """Call TTS API"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": "nova"
            }

            async with session.post(TTS_API, json=payload) as response:
                audio_data = await response.read()
                return audio_data
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice assistant"""
    await websocket.accept()
    logger.info("WebSocket connection established - WORKING VERSION")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                # Decode audio
                audio_bytes = base64.b64decode(data["data"])

                # Process through pipeline
                logger.info("Transcribing audio...")
                text = await transcribe_audio(audio_bytes)

                if text:
                    logger.info(f"Transcribed: {text}")
                    await websocket.send_json({"type": "transcription", "text": text})

                    logger.info("Generating response...")
                    response = await generate_response(text)

                    logger.info(f"Response: {response}")
                    await websocket.send_json({"type": "response", "text": response})

                    logger.info("Converting to speech...")
                    audio = await text_to_speech(response)

                    if audio:
                        audio_b64 = base64.b64encode(audio).decode('utf-8')
                        await websocket.send_json({"type": "audio_response", "data": audio_b64})
                        logger.info("Sent audio response")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/")
async def root():
    """Serve the web client"""
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "WORKING Voice Assistant"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting WORKING E2E Voice Assistant Server...")
    logger.info("This version bypasses Pipecat transport issues")
    logger.info("WebSocket endpoint: ws://localhost:8080/ws")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")