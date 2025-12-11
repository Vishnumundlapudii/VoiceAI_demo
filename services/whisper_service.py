"""
Custom Whisper ASR Service for Pipecat
Connects to E2E Networks Whisper API endpoint
"""

import asyncio
import aiohttp
import numpy as np
import wave
import io
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    ErrorFrame
)
from pipecat.services.ai_services import STTService
from pipecat.utils.audio import calculate_audio_volume

from loguru import logger


class WhisperHTTPService(STTService):
    """
    Custom Whisper service that sends audio to HTTP endpoint
    """

    def __init__(self, api_url: str, sample_rate: int = 16000):
        super().__init__()
        self._api_url = api_url
        self._sample_rate = sample_rate
        self._audio_buffer = bytearray()
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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio through Whisper API
        """
        try:
            # Convert audio bytes to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(audio)

            wav_data = wav_buffer.getvalue()

            # Send to Whisper API
            if not self._session:
                self._session = aiohttp.ClientSession()

            data = aiohttp.FormData()
            data.add_field('audio', wav_data,
                          filename='audio.wav',
                          content_type='audio/wav')

            async with self._session.post(
                self._api_url,
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    text = result.get('text', '').strip()

                    if text:
                        logger.info(f"Transcribed: {text}")
                        yield TranscriptionFrame(
                            text=text,
                            user_id="user",
                            timestamp=None
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"Whisper API error: {response.status} - {error_text}")
                    yield ErrorFrame(f"Transcription failed: {response.status}")

        except asyncio.TimeoutError:
            logger.error("Whisper API timeout")
            yield ErrorFrame("Transcription timeout")
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            yield ErrorFrame(f"Transcription error: {str(e)}")

    async def _process_audio_frame(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        """
        Buffer audio and process when we have enough
        """
        # Add to buffer
        self._audio_buffer.extend(frame.audio)

        # Process if we have enough audio (e.g., 1 second)
        buffer_duration = len(self._audio_buffer) / (self._sample_rate * 2)  # 2 bytes per sample

        if buffer_duration >= 1.0:
            # Process the buffered audio
            async for result_frame in self.run_stt(bytes(self._audio_buffer)):
                yield result_frame

            # Clear buffer
            self._audio_buffer.clear()

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """
        Process incoming frames
        """
        if isinstance(frame, AudioRawFrame):
            async for result in self._process_audio_frame(frame):
                yield result
        else:
            yield frame