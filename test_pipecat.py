#!/usr/bin/env python3
"""
Test script for True Pipecat Voice Assistant
Tests the pipeline with VAD and continuous audio processing
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from pipeline.voice_assistant import create_assistant
from loguru import logger
import config


async def test_pipecat_pipeline():
    """
    Test the voice assistant pipeline with VAD locally
    """
    try:
        logger.info("Starting Pipecat pipeline test with VAD...")
        logger.info("Speak naturally - VAD will detect when you start and stop speaking")

        # Create assistant
        assistant = create_assistant()

        # Create local audio transport with VAD
        transport = LocalAudioTransport(
            LocalAudioTransport.InputParams(
                sample_rate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        confidence_threshold=config.VAD_THRESHOLD,
                        silence_duration_ms=int(config.END_OF_SPEECH_THRESHOLD * 1000)
                    )
                )
            ),
            LocalAudioTransport.OutputParams(
                sample_rate=config.SAMPLE_RATE,
                channels=config.CHANNELS
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
    print("="*60)
    print("E2E Voice Assistant - Pipecat Pipeline Test")
    print("="*60)
    print("\nFeatures being tested:")
    print("✓ VAD (Voice Activity Detection)")
    print("✓ Continuous audio processing")
    print("✓ Pipeline frame processing")
    print("✓ Interruption handling")
    print("\nSpeak naturally - no push-to-talk needed!")
    print("Press Ctrl+C to stop.\n")

    asyncio.run(test_pipecat_pipeline())