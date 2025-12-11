"""
Working Pipecat WebSocket Server for Voice Assistant
Fixed implementation that actually works with Pipecat 0.0.36
"""

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Pipecat imports - using the working approach
from pipecat.frames.frames import AudioRawFrame, EndFrame, TranscriptionFrame, TextFrame
from pipecat.transports.network.websocket_server import WebsocketServerTransport
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from pipeline.voice_assistant import create_assistant
from loguru import logger
import config

app = FastAPI(title="E2E Voice Assistant - Working Pipecat")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WorkingPipecatHandler:
    """
    Working Pipecat WebSocket handler using WebsocketServerTransport
    """

    def __init__(self):
        self.active_connections = {}

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle WebSocket connection with working Pipecat approach
        """
        connection_id = id(websocket)

        try:
            await websocket.accept()
            logger.info(f"New WebSocket connection accepted: {connection_id}")

            # Create assistant pipeline
            assistant = create_assistant()

            # Store connection
            self.active_connections[connection_id] = {
                'websocket': websocket,
                'assistant': assistant
            }

            logger.info("Starting assistant pipeline...")

            # Create WebSocket server transport (this is the working approach)
            transport = WebsocketServerTransport(
                websocket=websocket,
                params={
                    "audio_out_enabled": True,
                    "audio_out_sample_rate": config.SAMPLE_RATE,
                    "audio_out_channels": config.CHANNELS
                }
            )

            # Run the pipeline with transport
            await assistant.run(transport)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error {connection_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                assistant = self.active_connections[connection_id].get('assistant')
                if assistant:
                    try:
                        await assistant.stop()
                    except:
                        pass
                del self.active_connections[connection_id]
            logger.info(f"Cleaned up connection: {connection_id}")


# Create global handler
ws_handler = WorkingPipecatHandler()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint using working Pipecat approach
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
        "service": "E2E Voice Assistant - Working Pipecat",
        "version": "2.1.0",
        "features": ["Pipeline", "VAD", "Services"]
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting E2E Voice Assistant - Working Pipecat Server...")
    logger.info("Features: Pipeline Processing, VAD, Frame Handling")
    logger.info("WebSocket endpoint: ws://localhost:8080/ws")
    logger.info("Web client: http://localhost:8080")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )