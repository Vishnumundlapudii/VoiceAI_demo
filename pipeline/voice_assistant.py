"""
Main Pipecat Pipeline for Voice Assistant
Orchestrates Whisper ASR → LLaMA LLM → Speech5 TTS
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame,
    EndFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_services import AIService
from pipecat.processors.filters.function_filter import FunctionFilter

from loguru import logger

# Import our custom services
from services.whisper_service import WhisperHTTPService
from services.llama_service import LLaMAHTTPService
from services.tts_service import Speech5HTTPService
import config


@dataclass
class AssistantConfig:
    """Configuration for the voice assistant"""
    whisper_url: str
    tts_url: str
    llama_base_url: str
    llama_token: str
    llama_model: str


class ConversationManager(FrameProcessor):
    """
    Manages conversation flow and context with interruption handling
    """

    def __init__(self):
        super().__init__()
        self.messages = []
        self.is_assistant_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Convert transcriptions to LLM messages and manage conversation with interruption handling
        """
        if isinstance(frame, TranscriptionFrame):
            # If assistant is speaking and user interrupts, handle it
            if self.is_assistant_speaking:
                logger.info(f"User interrupted: {frame.text}")
                # Could add interruption logic here
                self.is_assistant_speaking = False

            # Add user message to conversation
            user_message = {"role": "user", "content": frame.text}
            self.messages.append(user_message)

            logger.info(f"User said: {frame.text}")

            # Create LLM messages frame
            await self.push_frame(
                LLMMessagesFrame(messages=self.messages.copy()),
                direction
            )
        elif isinstance(frame, TextFrame):
            # This is an AI response - add to conversation history
            assistant_message = {"role": "assistant", "content": frame.text}
            self.messages.append(assistant_message)

            logger.info(f"Assistant responded: {frame.text}")

            # Mark that assistant is about to speak
            self.is_assistant_speaking = True

            # Pass the frame through
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame) and frame.num_channels > 0:
            # Audio output from TTS means assistant is speaking
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            # Assistant finished speaking
            self.is_assistant_speaking = False
            await self.push_frame(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)


class VoiceAssistantPipeline:
    """
    Main voice assistant pipeline using Pipecat
    """

    def __init__(self, config: AssistantConfig):
        self.config = config
        self._pipeline = None
        self._runner = None

    def create_pipeline(self) -> Pipeline:
        """
        Create the Pipecat pipeline
        """
        # Create services
        whisper_service = WhisperHTTPService(
            api_url=self.config.whisper_url,
            sample_rate=config.SAMPLE_RATE
        )

        llama_service = LLaMAHTTPService(
            api_key=self.config.llama_token,
            base_url=self.config.llama_base_url,
            model=self.config.llama_model
        )

        tts_service = Speech5HTTPService(
            api_url=self.config.tts_url,
            sample_rate=config.SAMPLE_RATE
        )

        # Create conversation manager
        conversation_manager = ConversationManager()

        # Create VAD analyzer
        vad_analyzer = SileroVADAnalyzer(
            params=VADParams(
                confidence_threshold=config.VAD_THRESHOLD,
                silence_duration_ms=int(config.END_OF_SPEECH_THRESHOLD * 1000)
            )
        )

        # Build pipeline with VAD
        pipeline = Pipeline([
            # VAD → Whisper
            vad_analyzer,
            whisper_service,

            # Transcription → Conversation Manager → LLaMA
            conversation_manager,
            llama_service,

            # LLaMA response → TTS
            tts_service,
        ])

        return pipeline

    async def run(self, transport):
        """
        Run the pipeline with given transport
        """
        try:
            logger.info("Starting voice assistant pipeline...")

            # Create pipeline
            self._pipeline = self.create_pipeline()

            # Create runner
            self._runner = PipelineRunner()

            # Create task
            task = PipelineTask(
                self._pipeline,
                PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True
                )
            )

            # Run pipeline
            await self._runner.run(task, transport)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    async def stop(self):
        """
        Stop the pipeline
        """
        if self._runner:
            await self._runner.stop()


# Helper function to create assistant
def create_assistant() -> VoiceAssistantPipeline:
    """
    Create voice assistant with default configuration
    """
    assistant_config = AssistantConfig(
        whisper_url=config.WHISPER_API,
        tts_url=config.TTS_API,
        llama_base_url=config.LLAMA_BASE_URL,
        llama_token=config.E2E_TOKEN,
        llama_model=config.LLAMA_MODEL
    )

    return VoiceAssistantPipeline(assistant_config)