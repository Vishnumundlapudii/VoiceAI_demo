# E2E Voice Assistant with Pipecat

A real-time voice assistant built using the **Pipecat framework**, orchestrating open-source AI models through E2E Networks infrastructure.

## 🎯 Overview

This project demonstrates a production-ready voice assistant pipeline using:
- **Whisper ASR** for speech-to-text
- **LLaMA 3.3 70B** for intelligent responses
- **Speech5 TTS** for text-to-speech
- **Pipecat Framework** for real-time pipeline orchestration

## 🏗️ Architecture

```
User Voice → WebSocket → Pipecat Pipeline → AI Response
                             ↓
                      [Whisper ASR API]
                             ↓
                      [LLaMA 3.3 70B API]
                             ↓
                      [Speech5 TTS API]
                             ↓
                      [Audio Response]
```

## 📁 Project Structure

```
phase_2/
├── services/               # Custom service adapters
│   ├── whisper_service.py  # Whisper ASR adapter
│   ├── llama_service.py    # LLaMA LLM adapter
│   └── tts_service.py      # Speech5 TTS adapter
├── pipeline/
│   └── voice_assistant.py  # Main Pipecat pipeline
├── web/
│   └── index.html          # Web interface
├── config.py               # Configuration
├── server.py               # WebSocket server
├── test_local.py           # Local testing
├── requirements.txt        # Dependencies
└── setup.sh                # Setup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Access to E2E Networks API endpoints
- Microphone and speakers for testing

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd e2e-voice-assistant
```

2. Run setup script:
```bash
bash setup.sh
```

3. Configure your API endpoints:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual endpoints and token
nano .env  # or use any text editor
```

Fill in your actual values:
```env
WHISPER_API=http://your-whisper-endpoint:8000/transcribe
TTS_API=http://your-tts-endpoint:8000/v1/audio/speech
LLAMA_BASE_URL=https://your-llama-endpoint/v1
E2E_TOKEN=your-actual-token-here
```

### Running the Assistant

1. **Start the WebSocket server:**
```bash
source venv/bin/activate
python server.py
```

2. **Open the web interface:**
Navigate to `http://localhost:8080` in your browser

3. **Use the assistant:**
   - Click "Connect" to establish connection
   - Hold the microphone button to speak
   - Release to send your message
   - Listen to the AI response

### Local Testing

Test the pipeline without WebSocket:
```bash
python test_local.py
```

## 🔧 Configuration

Edit `config.py` to customize:
- API endpoints
- Model parameters
- Audio settings
- VAD thresholds

## 📚 How It Works

### 1. Custom Service Adapters
We created Pipecat-compatible adapters for each E2E Networks endpoint:
- `WhisperHTTPService`: Handles ASR via HTTP API
- `LLaMAHTTPService`: Manages LLM via OpenAI-compatible API
- `Speech5HTTPService`: Generates TTS via HTTP API

### 2. Pipeline Orchestration
The Pipecat pipeline manages:
- Audio frame processing
- Voice activity detection (VAD)
- Service coordination
- Interruption handling
- Real-time streaming

### 3. WebSocket Transport
- Real-time bidirectional communication
- Browser-based audio capture
- Low-latency streaming

## 🛠️ Development

### Adding New Features

1. **Custom processors:** Create new frame processors in `pipeline/`
2. **Service adapters:** Add new services in `services/`
3. **Transport layers:** Implement new transports (WebRTC, etc.)

### Testing Services

Test individual services:
```python
from services.whisper_service import WhisperHTTPService
# Test your service...
```

## 📊 Performance

- **Latency Target:** < 500ms total round-trip
- **Audio Format:** 16kHz, mono, 16-bit PCM
- **Supported Browsers:** Chrome, Firefox, Safari, Edge

## 🔍 Troubleshooting

### Connection Issues
- Verify all API endpoints are accessible
- Check token validity
- Ensure proper CORS configuration

### Audio Issues
- Grant microphone permissions
- Check audio device settings
- Verify sample rate compatibility

### Pipeline Issues
- Check service logs for errors
- Verify frame processing chain
- Monitor WebSocket messages

## 📝 API Reference

### WebSocket Messages

**Client → Server:**
```json
{
  "type": "audio",
  "data": [/* PCM16 audio samples */]
}
```

**Server → Client:**
```json
{
  "type": "transcription",
  "text": "User speech text"
}
```

```json
{
  "type": "response",
  "text": "AI response text"
}
```

```json
{
  "type": "audio",
  "data": [/* PCM16 audio samples */]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **Pipecat Framework** - Real-time AI pipeline orchestration
- **E2E Networks** - AI infrastructure and model hosting
- **Open-source Models** - Whisper, LLaMA, Speech5

## 📧 Contact

For questions or support, please contact [your-email]

---

Built with ❤️ using open-source AI models and the Pipecat framework