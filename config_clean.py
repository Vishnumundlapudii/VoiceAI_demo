"""
UNIFIED VOICE ASSISTANT CONFIGURATION
All parameters in one place - no more scattered values!
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API ENDPOINTS & AUTHENTICATION
# =============================================================================

WHISPER_API = os.getenv("WHISPER_API", "http://your-whisper-endpoint:8000/transcribe")
TTS_API = os.getenv("TTS_API", "http://216.48.191.105:8000/v1/audio/speech")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "https://your-llama-endpoint/v1")
E2E_TOKEN = os.getenv("E2E_TOKEN", "your-token-here")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000        # Audio sample rate (Hz)
CHANNELS = 1               # Mono audio
AUDIO_BIT_DEPTH = 16       # 16-bit audio

# =============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# =============================================================================

# Simple VAD energy threshold - the core parameter for speech detection
VAD_ENERGY_THRESHOLD = 3000000      # Set below your actual speech energy

# Speech timing parameters
END_OF_SPEECH_THRESHOLD = 2.5       # More time to finish speaking before processing
SPEECH_TIMEOUT_THRESHOLD = 15.0     # Max speech duration before timeout

# Minimum audio requirements
MIN_AUDIO_DURATION = 1.0             # Require longer speech for better Whisper accuracy

# =============================================================================
# LANGUAGE MODEL (LLM) CONFIGURATION
# =============================================================================

LLM_MAX_TOKENS = 150                 # Response length limit
LLM_TEMPERATURE = 0.7                # Creativity level (0.0-1.0)
LLM_TOP_P = 0.9                      # Token selection focus
LLM_PRESENCE_PENALTY = 0.1           # Encourage new topics
LLM_FREQUENCY_PENALTY = 0.1          # Reduce repetition

# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

CONVERSATION_CONTEXT_LENGTH = 10     # Number of messages to remember
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise."""

# =============================================================================
# NETWORK & TIMEOUTS
# =============================================================================

WHISPER_TIMEOUT = 30                 # Whisper API timeout (seconds)
LLM_TIMEOUT = 30                     # LLM API timeout (seconds)
TTS_TIMEOUT = 10                     # TTS API timeout (seconds)

# =============================================================================
# DEVELOPMENT & DEBUGGING
# =============================================================================

DEBUG_VAD = False                    # Enable VAD debug logging
DEBUG_AUDIO = False                  # Enable audio debug logging
DEBUG_LLM = False                    # Enable LLM debug logging

# =============================================================================
# DISPLAY CONFIGURATION (for verification)
# =============================================================================

if __name__ == "__main__":
    print("🔧 UNIFIED VOICE ASSISTANT CONFIGURATION")
    print("=" * 50)

    print("\n📡 API ENDPOINTS:")
    print(f"  WHISPER: {WHISPER_API[:50]}...")
    print(f"  TTS:     {TTS_API[:50]}...")
    print(f"  LLAMA:   {LLAMA_BASE_URL[:50]}...")
    print(f"  TOKEN:   {'*' * 20} (hidden)")
    print(f"  MODEL:   {LLAMA_MODEL}")

    print("\n🎵 AUDIO CONFIG:")
    print(f"  Sample Rate:      {SAMPLE_RATE} Hz")
    print(f"  Channels:         {CHANNELS} (Mono)")
    print(f"  Bit Depth:        {AUDIO_BIT_DEPTH}-bit")

    print("\n🗣️ VAD CONFIG:")
    print(f"  Energy Threshold: {VAD_ENERGY_THRESHOLD:,}")
    print(f"  Speech End Time:  {END_OF_SPEECH_THRESHOLD}s")
    print(f"  Speech Timeout:   {SPEECH_TIMEOUT_THRESHOLD}s")

    print("\n🧠 LLM CONFIG:")
    print(f"  Max Tokens:       {LLM_MAX_TOKENS}")
    print(f"  Temperature:      {LLM_TEMPERATURE}")
    print(f"  Context Length:   {CONVERSATION_CONTEXT_LENGTH} messages")

    print("\n⏱️ TIMEOUTS:")
    print(f"  Whisper:          {WHISPER_TIMEOUT}s")
    print(f"  LLM:              {LLM_TIMEOUT}s")
    print(f"  TTS:              {TTS_TIMEOUT}s")

    print("\n" + "=" * 50)
    print("✅ Configuration loaded successfully!")