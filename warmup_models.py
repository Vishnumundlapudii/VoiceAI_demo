#!/usr/bin/env python3
"""
Model Warmup Script
Pre-loads all AI models to eliminate cold start delays
Run this before demos to ensure instant responses
"""

import asyncio
import aiohttp
import json
import base64
import io
import wave
import numpy as np
from openai import AsyncOpenAI
import time
from loguru import logger
import config

class ModelWarmer:
    def __init__(self):
        self.results = {}

    async def create_test_audio(self):
        """Create a small test audio file"""
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A note

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()

    async def warmup_whisper(self):
        """Warmup Whisper ASR model"""
        logger.info("🎤 Warming up Whisper...")
        start_time = time.time()

        try:
            test_audio = await self.create_test_audio()

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                data = aiohttp.FormData()
                data.add_field('audio', test_audio, filename='warmup.wav', content_type='audio/wav')

                async with session.post(config.WHISPER_API, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        warmup_time = time.time() - start_time
                        logger.info(f"✅ Whisper warmed up in {warmup_time:.2f}s")
                        self.results['whisper'] = {
                            'status': 'success',
                            'time': warmup_time,
                            'response': result.get('text', '')
                        }
                        return True
                    else:
                        logger.error(f"❌ Whisper warmup failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Whisper warmup error: {e}")

        self.results['whisper'] = {'status': 'failed', 'time': time.time() - start_time}
        return False

    async def warmup_llama(self):
        """Warmup LLaMA model"""
        logger.info("🧠 Warming up LLaMA...")
        start_time = time.time()

        try:
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL,
                timeout=30.0
            )

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                temperature=0.7
            )

            warmup_time = time.time() - start_time
            response_text = response.choices[0].message.content
            logger.info(f"✅ LLaMA warmed up in {warmup_time:.2f}s")
            self.results['llama'] = {
                'status': 'success',
                'time': warmup_time,
                'response': response_text
            }
            return True

        except Exception as e:
            logger.error(f"❌ LLaMA warmup error: {e}")

        self.results['llama'] = {'status': 'failed', 'time': time.time() - start_time}
        return False

    async def warmup_tts(self):
        """Warmup Glow-TTS model"""
        logger.info("🔊 Warming up Glow-TTS...")
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                payload = {"input": "Hello"}
                headers = {"Content-Type": "application/json"}
                glow_tts_url = "http://216.48.191.105:8000/v1/audio/speech"

                async with session.post(glow_tts_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        warmup_time = time.time() - start_time
                        logger.info(f"✅ Glow-TTS warmed up in {warmup_time:.2f}s")
                        self.results['tts'] = {
                            'status': 'success',
                            'time': warmup_time,
                            'audio_size': len(result.get('audio', ''))
                        }
                        return True
                    else:
                        logger.error(f"❌ TTS warmup failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ TTS warmup error: {e}")

        self.results['tts'] = {'status': 'failed', 'time': time.time() - start_time}
        return False

    async def warmup_all(self):
        """Warmup all models in parallel"""
        logger.info("🚀 Starting model warmup...")
        total_start = time.time()

        # Run all warmups in parallel
        whisper_task = asyncio.create_task(self.warmup_whisper())
        llama_task = asyncio.create_task(self.warmup_llama())
        tts_task = asyncio.create_task(self.warmup_tts())

        # Wait for all to complete
        results = await asyncio.gather(whisper_task, llama_task, tts_task, return_exceptions=True)

        total_time = time.time() - total_start

        # Summary
        logger.info("📊 Warmup Summary:")
        logger.info("=" * 50)

        success_count = 0
        for service, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            logger.info(f"{status_icon} {service.upper()}: {result['time']:.2f}s")
            if result['status'] == 'success':
                success_count += 1

        logger.info("=" * 50)
        logger.info(f"🎯 Total warmup time: {total_time:.2f}s")
        logger.info(f"🎯 Services ready: {success_count}/3")

        if success_count == 3:
            logger.info("🎉 All models warmed up! Ready for instant responses!")
            return True
        else:
            logger.warning(f"⚠️ Only {success_count}/3 services ready")
            return False

    def get_warmup_summary(self):
        """Get warmup results summary"""
        return self.results

async def main():
    """Main warmup function"""
    warmer = ModelWarmer()
    success = await warmer.warmup_all()

    if success:
        print("\n🎉 SUCCESS: All models warmed up!")
        print("🚀 Your voice assistant is now ready for instant responses!")
    else:
        print("\n⚠️ WARNING: Some models failed to warm up")
        print("💡 Check the logs above for details")

    return success

if __name__ == "__main__":
    asyncio.run(main())