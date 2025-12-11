"""
True Pipecat WebSocket Server for Voice Assistant
Uses Pipecat's transport layer and VAD
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Pipecat imports for 0.0.36
from pipecat.frames.frames import AudioRawFrame, EndFrame
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

from pipeline.voice_assistant import create_assistant
from loguru import logger
import config

app = FastAPI(title="E2E Voice Assistant - True Pipecat")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipecatWebSocketHandler:
    """
    Handles WebSocket connections using Pipecat's transport
    """

    def __init__(self):
        self.active_connections = {}

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle WebSocket connection with Pipecat transport
        """
        connection_id = id(websocket)

        try:
            logger.info(f"New WebSocket connection: {connection_id}")

            # Create Pipecat transport with default parameters
            transport = FastAPIWebsocketTransport(websocket=websocket)

            # Store connection
            self.active_connections[connection_id] = {
                'websocket': websocket,
                'transport': transport,
                'assistant': None
            }

            # Create and run assistant
            assistant = create_assistant()
            self.active_connections[connection_id]['assistant'] = assistant

            # Run the pipeline
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
                    await assistant.stop()
                del self.active_connections[connection_id]
            logger.info(f"Cleaned up connection: {connection_id}")


# Create global handler
ws_handler = PipecatWebSocketHandler()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint using Pipecat transport
    """
    await ws_handler.handle_connection(websocket)


@app.get("/")
async def root():
    """
    Serve the Pipecat web client
    """
    with open("web/index_pipecat.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "E2E Voice Assistant - True Pipecat",
        "version": "2.0.0",
        "features": ["VAD", "Interruption", "Pipeline"]
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting E2E Voice Assistant - True Pipecat Server...")
    logger.info("Features: VAD, Interruption Handling, Pipeline Processing")
    logger.info("WebSocket endpoint: ws://localhost:8080/ws")
    logger.info("Web client: http://localhost:8080")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )