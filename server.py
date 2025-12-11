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
        Handle a WebSocket connection
        """
        await websocket.accept()
        logger.info("WebSocket connection established")

        try:
            # Create assistant pipeline
            self.assistant = create_assistant()

            # Create WebSocket transport for Pipecat
            # For Pipecat 0.0.36, use FastAPIWebsocketTransport with minimal params
            from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams

            # Use default params - let Pipecat handle everything
            params = FastAPIWebsocketParams()

            self.transport = FastAPIWebsocketTransport(
                websocket=websocket,
                params=params
            )

            # Run the pipeline with transport
            await self.assistant.run(self.transport)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.assistant:
                await self.assistant.stop()


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