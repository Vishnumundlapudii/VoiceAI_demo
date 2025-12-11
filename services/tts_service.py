"""
Custom Speech5 TTS Service for Pipecat
Connects to E2E Networks TTS API endpoint
"""

import asyncio
import aiohttp
import base64
import io
import wave
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    ErrorFrame
)
from pipecat.services.ai_services import TTSService
from pipecat.processors.frame_processor import FrameDirection

from loguru import logger


class Speech5HTTPService(TTSService):
    """
    Custom TTS service that sends text to Speech5 HTTP endpoint
    """

    def __init__(self, api_url: str, voice: str = "default", sample_rate: int = 16000):
        super().__init__(sample_rate=sample_rate)
        self._api_url = api_url
        self._voice = voice
        self._session = None

    async def start(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Initialize the service"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        yield frame

    async def stop(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Cleanup the service"""
        if self._session:
            await self._session.close()
            self._session = None
        yield frame

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Convert text to speech using Speech5 API
        """
        try:
            logger.info(f"Generating speech for: {text[:50]}...")

            if not self._session:
                self._session = aiohttp.ClientSession()

            # Prepare request payload
            payload = {
                "model": "speecht5_tts",
                "input": text,
                "voice": self._voice,
                "response_format": "wav"
            }

            # Send to TTS API
            async with self._session.post(
                self._api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    audio_base64 = result.get('audio')

                    if audio_base64:
                        # Decode base64 audio
                        audio_data = base64.b64decode(audio_base64)

                        # Parse WAV to get raw audio
                        wav_buffer = io.BytesIO(audio_data)
                        with wave.open(wav_buffer, 'rb') as wav_file:
                            frames = wav_file.readframes(wav_file.getnframes())

                        # Create audio frame
                        # Pipecat expects raw PCM audio
                        yield AudioRawFrame(
                            audio=frames,
                            sample_rate=self._sample_rate,
                            num_channels=1
                        )

                        logger.info(f"Generated {len(frames)} bytes of audio")
                else:
                    error_text = await response.text()
                    logger.error(f"TTS API error: {response.status} - {error_text}")
                    yield ErrorFrame(f"TTS failed: {response.status}")

        except asyncio.TimeoutError:
            logger.error("TTS API timeout")
            yield ErrorFrame("TTS timeout")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            yield ErrorFrame(f"TTS error: {str(e)}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> AsyncGenerator[Frame, None]:
        """
        Process incoming frames - Updated for Pipecat 0.0.36
        """
        if isinstance(frame, TextFrame):
            # Convert text to speech
            async for audio_frame in self.run_tts(frame.text):
                yield audio_frame
        else:
            # Pass through other frames
            yield frame