"""
Step 1: Basic Pipecat Pipeline - Understanding Frame Processing
This shows how Pipecat processes data through frames
"""

import asyncio
from loguru import logger

# Basic Pipecat imports
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

# Import our custom services
from services.whisper_service import WhisperHTTPService
from services.llama_service import LLaMAHTTPService
from services.tts_service import Speech5HTTPService
import config

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

class SimpleFrameLogger(FrameProcessor):
    """
    Simple processor to log what frames are flowing through the pipeline
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Log each frame that passes through"""
        frame_type = type(frame).__name__
        logger.info(f"[{self.name}] Processing frame: {frame_type}")

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[{self.name}] Transcription: {frame.text}")
        elif isinstance(frame, TextFrame):
            logger.info(f"[{self.name}] Text: {frame.text}")

        # Return the frame unchanged
        yield frame

async def test_basic_pipeline():
    """
    Test the basic pipeline with your E2E services
    """
    logger.info("=== Step 1: Testing Basic Pipecat Pipeline ===")

    # Create your E2E services
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

    # Create frame loggers to see what's happening
    logger1 = SimpleFrameLogger("AFTER-WHISPER")
    logger2 = SimpleFrameLogger("AFTER-LLAMA")
    logger3 = SimpleFrameLogger("AFTER-TTS")

    # Build the pipeline
    pipeline = Pipeline([
        whisper_service,
        logger1,
        llama_service,
        logger2,
        tts_service,
        logger3
    ])

    logger.info("✅ Basic pipeline created successfully!")
    logger.info("Pipeline flow: Audio → Whisper → LLaMA → TTS")
    logger.info("Next step: Add WebSocket transport")

if __name__ == "__main__":
    asyncio.run(test_basic_pipeline())