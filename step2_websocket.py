"""
Step 2: Add WebSocket Transport - Connect Browser to Pipeline
This shows how to connect a WebSocket to the Pipecat pipeline
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Pipecat WebSocket transport
from pipecat.transports.network.websocket_server import WebsocketServerTransport
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

# Import our services
from services.whisper_service import WhisperHTTPService
from services.llama_service import LLaMAHTTPService
from services.tts_service import Speech5HTTPService
import config

app = FastAPI(title="Step 2: Pipecat WebSocket")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Step2Handler:
    """
    Simple WebSocket handler that connects browser to Pipecat pipeline
    """

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("✅ Step 2: WebSocket connected")

        try:
            # Create the pipeline
            pipeline = self.create_pipeline()
            logger.info("✅ Pipeline created")

            # Create WebSocket transport with correct parameters
            transport = WebsocketServerTransport(
                host="0.0.0.0",
                port=8080,
                websocket=websocket
            )
            logger.info("✅ Transport created")

            # Create pipeline runner
            runner = PipelineRunner()
            task = PipelineTask(
                pipeline,
                PipelineParams(
                    allow_interruptions=False,  # Start simple
                    enable_metrics=False
                )
            )

            logger.info("🚀 Starting pipeline...")
            await runner.run(task, transport)

        except WebSocketDisconnect:
            logger.info("❌ WebSocket disconnected")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def create_pipeline(self):
        """Create basic pipeline with your E2E services"""
        # Your E2E services
        whisper_service = WhisperHTTPService(
            api_url=config.WHISPER_API,
            sample_rate=config.SAMPLE_RATE
        )

        llama_service = LLaMAHTTPService(
            api_key=config.E2E_TOKEN,
            base_url=config.LLAMA_BASE_URL,
            model=config.LLAMA_MODEL
        )

        tts_service = Speech5HTTPService(
            api_url=config.TTS_API,
            sample_rate=config.SAMPLE_RATE
        )

        # Simple pipeline
        pipeline = Pipeline([
            whisper_service,
            llama_service,
            tts_service,
        ])

        return pipeline

# Global handler
handler = Step2Handler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Step 2: Basic WebSocket endpoint"""
    await handler.handle_connection(websocket)

@app.get("/")
async def root():
    """Serve your existing web client"""
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"step": 2, "features": ["Pipeline", "WebSocket", "E2E Services"]}

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Step 2: Starting Pipecat with WebSocket")
    logger.info("Features: Basic Pipeline + WebSocket Transport")
    logger.info("Next: Add VAD for automatic voice detection")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")