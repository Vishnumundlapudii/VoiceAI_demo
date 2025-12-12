"""
Configuration for Phase 2 Pipecat Demo
Loads sensitive data from environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Endpoints (loaded from environment or defaults for GitHub)
WHISPER_API = os.getenv("WHISPER_API", "http://your-whisper-endpoint:8000/transcribe")
TTS_API = os.getenv("TTS_API", "http://your-tts-endpoint:8000/v1/audio/speech")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "https://your-llama-endpoint/v1")

# E2E Networks Token
E2E_TOKEN = os.getenv("E2E_TOKEN", "your-token-here")

# Model Configuration
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1

# Pipeline Settings - BALANCED (Fixed)
VAD_THRESHOLD = 0.3  # Lower threshold for better sensitivity
END_OF_SPEECH_THRESHOLD = 1.5  # SAFER: Balanced timing
SPEECH_TIMEOUT_THRESHOLD = 8.0  # Maximum continuous speech duration
FAST_MODE_THRESHOLD = 1.2  # More conservative fast mode

# Display loaded configuration (for debugging)
if __name__ == "__main__":
    print("Configuration Loaded:")
    print(f"WHISPER_API: {WHISPER_API[:30]}..." if len(WHISPER_API) > 30 else f"WHISPER_API: {WHISPER_API}")
    print(f"TTS_API: {TTS_API[:30]}..." if len(TTS_API) > 30 else f"TTS_API: {TTS_API}")
    print(f"LLAMA_BASE_URL: {LLAMA_BASE_URL[:30]}..." if len(LLAMA_BASE_URL) > 30 else f"LLAMA_BASE_URL: {LLAMA_BASE_URL}")
    print(f"E2E_TOKEN: {'*' * 10} (hidden)")
    print(f"LLAMA_MODEL: {LLAMA_MODEL}")