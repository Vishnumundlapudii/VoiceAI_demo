"""
WebSocket Server for Pipecat Voice Assistant
Handles real-time audio streaming
"""

import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Pipecat imports for 0.0.36
from pipecat.frames.frames import AudioRawFrame, EndFrame
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from pipeline.voice_assistant import create_assistant
from loguru import logger

app = FastAPI(title="E2E Voice Assistant - Pipecat")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketHandler:
    """
    Handles WebSocket connections for voice assistant
    """

    def __init__(self):
        self.assistant = None
        self.transport = None

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle a WebSocket connection - Direct API approach
        """
        await websocket.accept()
        logger.info("WebSocket connection established")

        try:
            while True:
                # Receive JSON message from client
                data = await websocket.receive_json()

                if data.get("type") == "audio":
                    # Process the audio through our pipeline
                    await self.process_audio(websocket, data["data"])
                elif data.get("type") == "end_of_speech":
                    # Handle end of speech if needed
                    pass

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def process_audio(self, websocket: WebSocket, base64_audio: str):
        """Process audio through our pipeline"""
        try:
            import base64
            import aiohttp
            from openai import AsyncOpenAI
            import config

            # Decode audio
            audio_bytes = base64.b64decode(base64_audio)

            # 1. Transcribe with Whisper (faster-whisper format)
            logger.info("Transcribing audio...")
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                # Your Whisper API expects 'audio' field name
                data.add_field('audio', audio_bytes, filename='recording.wav', content_type='audio/wav')

                async with session.post(config.WHISPER_API, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get('text', '').strip()
                        logger.info(f"Whisper response: {result}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Whisper API error {response.status}: {error_text}")
                        await websocket.send_json({"type": "error", "text": f"Whisper API error: {response.status}"})
                        return

            if not text:
                return

            logger.info(f"Transcribed: {text}")
            await websocket.send_json({"type": "transcription", "text": text})

            # 2. Generate response with LLaMA
            logger.info("Generating response...")
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL
            )

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100
            )

            response_text = response.choices[0].message.content
            logger.info(f"Response: {response_text}")
            await websocket.send_json({"type": "response", "text": response_text})

            # 3. Convert to speech with TTS
            logger.info("Converting to speech...")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "tts-1",
                    "input": response_text,
                    "voice": "nova"
                }

                async with session.post(config.TTS_API, json=payload) as tts_response:
                    audio_data = await tts_response.read()

                    if audio_data:
                        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                        await websocket.send_json({"type": "audio_response", "data": audio_b64})
                        logger.info("Sent audio response")

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await websocket.send_json({"type": "error", "text": str(e)})


# Create global handler
ws_handler = WebSocketHandler()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice assistant
    """
    await ws_handler.handle_connection(websocket)


@app.get("/")
async def root():
    """
    Serve the web client
    """
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "E2E Voice Assistant - Pipecat",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting E2E Voice Assistant Server...")
    logger.info("WebSocket endpoint: ws://localhost:8080/ws")
    logger.info("Web client: http://localhost:8080")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )