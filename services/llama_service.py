"""
Custom LLaMA LLM Service for Pipecat
Connects to E2E Networks LLaMA endpoint via OpenAI client
"""

import asyncio
from typing import AsyncGenerator, List
from openai import AsyncOpenAI

from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    ErrorFrame
)
from pipecat.services.ai_services import LLMService
from pipecat.processors.frame_processor import FrameDirection

from loguru import logger


class LLaMAHTTPService(LLMService):
    """
    Custom LLaMA service using E2E Networks endpoint
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "meta-llama/Llama-3.3-70B-Instruct"
    ):
        super().__init__()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self._model = model

    async def _generate_response(self, messages: List[dict]) -> AsyncGenerator[Frame, None]:
        """
        Generate response from LLaMA
        """
        try:
            logger.info(f"Sending to LLaMA: {messages[-1]['content']}")

            # Add system message if not present
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Be conversational and concise."
                })

            # Call LLaMA API
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                stream=False  # Start with non-streaming for simplicity
            )

            # Extract response
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"LLaMA response: {ai_response}")

            # Send response frames
            yield LLMFullResponseStartFrame()
            yield TextFrame(text=ai_response)
            yield LLMFullResponseEndFrame()

        except Exception as e:
            logger.error(f"LLaMA error: {e}")
            yield ErrorFrame(f"LLM error: {str(e)}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> AsyncGenerator[Frame, None]:
        """
        Process incoming frames - Updated for Pipecat 0.0.36
        """
        if isinstance(frame, LLMMessagesFrame):
            # Process LLM messages
            async for response_frame in self._generate_response(frame.messages):
                yield response_frame
        else:
            # Pass through other frames
            yield frame