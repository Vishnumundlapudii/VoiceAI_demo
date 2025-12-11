#!/usr/bin/env python3
"""
Local test script for Pipecat Voice Assistant
Tests the pipeline with local audio input/output
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.local.audio import LocalAudioTransport

from pipeline.voice_assistant import create_assistant
from loguru import logger


async def test_pipeline():
    """
    Test the voice assistant pipeline locally
    """
    try:
        logger.info("Starting local pipeline test...")
        logger.info("Speak into your microphone when ready...")

        # Create assistant
        assistant = create_assistant()

        # Create local audio transport
        transport = LocalAudioTransport(
            LocalAudioTransport.InputParams(
                sample_rate=16000,
                channels=1
            ),
            LocalAudioTransport.OutputParams(
                sample_rate=16000,
                channels=1
            )
        )

        # Run the pipeline
        await assistant.run(transport)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*50)
    print("E2E Voice Assistant - Local Test")
    print("="*50)
    print("\nThis will test the pipeline with your microphone and speakers.")
    print("Press Ctrl+C to stop.\n")

    asyncio.run(test_pipeline())