#!/usr/bin/env python3
"""
Quick Whisper API Test
Tests if your Whisper server is working correctly
"""

import requests
import io
import wave
import numpy as np

# Generate a test audio file (simple sine wave that says nothing)
def create_test_audio():
    sample_rate = 16000
    duration = 2  # 2 seconds
    frequency = 440  # A note

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3

    # Convert to 16-bit integers
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return wav_buffer.getvalue()

def test_whisper_api():
    print("🧪 Testing Whisper API...")

    whisper_url = "http://101.53.140.94:8000/transcribe"

    # Create test audio
    test_audio = create_test_audio()
    print(f"📊 Generated test audio: {len(test_audio)} bytes")

    try:
        # Test the API
        files = {'audio': ('test.wav', test_audio, 'audio/wav')}
        response = requests.post(whisper_url, files=files, timeout=10)

        print(f"📡 Response status: {response.status_code}")
        print(f"📝 Response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Whisper API working!")
            print(f"🎯 Transcription: '{result.get('text', 'NO TEXT')}'")
        else:
            print(f"❌ API Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_whisper_api()