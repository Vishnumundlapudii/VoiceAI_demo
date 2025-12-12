"""
Production-Ready Enhanced Voice Assistant Server
Integrates advanced VAD, audio processing, error handling, and monitoring
for production deployment.
"""

import asyncio
import json
import base64
import time
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
from openai import AsyncOpenAI
import traceback
from contextlib import asynccontextmanager

# Import our enhanced modules
from enhanced_vad import EnhancedVAD, VADConfig
from audio_processor import AudioProcessor, AudioConfig, InterruptionHandler

from loguru import logger
import config

# Production monitoring and health
from dataclasses import dataclass
from collections import defaultdict, deque
import psutil
import sys

@dataclass
class ServiceHealth:
    """Service health tracking"""
    whisper_status: str = "unknown"
    llama_status: str = "unknown"
    tts_status: str = "unknown"
    last_check: float = 0
    error_count: int = 0
    total_requests: int = 0

@dataclass
class SessionMetrics:
    """Per-session metrics tracking"""
    session_id: str
    start_time: float
    total_audio_processed: int = 0
    successful_completions: int = 0
    errors: int = 0
    interruptions: int = 0
    avg_response_time: float = 0.0

# Global health and metrics
service_health = ServiceHealth()
session_metrics: Dict[str, SessionMetrics] = {}
recent_errors = deque(maxlen=100)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Production Voice Assistant Server")

    # Initial health check
    await check_service_health()

    yield

    logger.info("🛑 Shutting down Production Voice Assistant Server")

app = FastAPI(
    title="E2E Voice Assistant - Production",
    version="3.0.0",
    lifespan=lifespan
)

# Enhanced CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ProductionVoiceHandler:
    """
    Production-ready voice handler with enhanced features
    """

    def __init__(self):
        # Enhanced VAD with production config
        vad_config = VADConfig(
            min_speech_duration_ms=80,      # Faster response
            min_silence_duration_ms=400,    # Quicker end-of-speech detection
            adaptation_rate=0.15,           # Faster adaptation
        )
        self.vad = EnhancedVAD(
            sample_rate=config.SAMPLE_RATE,
            config=vad_config
        )

        # Enhanced audio processing
        audio_config = AudioConfig(
            input_sample_rate=config.SAMPLE_RATE,
            output_sample_rate=22050,       # Higher quality output
            chunk_size_ms=30,               # Smaller chunks for faster response
            enable_noise_reduction=True,
            enable_agc=True,
            enable_compressor=True
        )
        self.audio_processor = AudioProcessor(config=audio_config)

        # Fast interruption handling
        self.interruption_handler = InterruptionHandler(sensitivity=0.8)

        # Session management with metrics
        self.active_sessions: Dict[str, dict] = {}

        # Performance monitoring
        self.request_times = deque(maxlen=1000)

        logger.info("✅ Production Voice Handler initialized")

    async def handle_connection(self, websocket: WebSocket):
        """Enhanced WebSocket connection handling with monitoring"""
        session_id = str(id(websocket))
        start_time = time.time()

        try:
            await websocket.accept()
            logger.info(f"✅ Production WebSocket connected: {session_id}")

            # Initialize session with metrics
            self.active_sessions[session_id] = {
                'websocket': websocket,
                'audio_buffer': bytearray(),
                'is_speaking': False,
                'assistant_speaking': False,
                'last_speech_time': None,
                'conversation_context': [],
                'session_start': start_time,
                'request_count': 0,
                'error_count': 0,
                'last_activity': start_time,
                'vad_state': {},
                'audio_stats': {}
            }

            # Track session metrics
            session_metrics[session_id] = SessionMetrics(
                session_id=session_id,
                start_time=start_time
            )

            # Reset processors for new session
            self.vad.reset()
            self.audio_processor.reset()

            await websocket.send_json({
                "type": "connection_status",
                "status": "connected",
                "session_id": session_id,
                "features": [
                    "Enhanced VAD",
                    "Adaptive Thresholds",
                    "Noise Reduction",
                    "Fast Interruption",
                    "Audio Enhancement",
                    "Production Monitoring"
                ],
                "message": "🎤 Production-ready voice assistant active!"
            })

            # Send initial service health
            await websocket.send_json({
                "type": "service_health",
                "health": await self.get_service_status()
            })

            # Main message loop with enhanced error handling
            while True:
                try:
                    # Set timeout for WebSocket operations
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=30.0  # 30 second timeout
                    )

                    # Update activity timestamp
                    self.active_sessions[session_id]['last_activity'] = time.time()

                    if data.get("type") == "audio_chunk":
                        await self.process_audio_chunk(session_id, data["data"])
                    elif data.get("type") == "audio":  # Legacy support
                        await self.process_legacy_audio(session_id, data["data"])
                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})

                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Session {session_id} timeout - checking connection")
                    await websocket.send_json({
                        "type": "ping",
                        "message": "Connection check"
                    })

        except WebSocketDisconnect:
            logger.info(f"❌ Production WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Production WebSocket error {session_id}: {e}")
            logger.error(traceback.format_exc())

            # Track error
            if session_id in session_metrics:
                session_metrics[session_id].errors += 1

            recent_errors.append({
                'session_id': session_id,
                'error': str(e),
                'timestamp': time.time(),
                'traceback': traceback.format_exc()
            })

        finally:
            # Cleanup with metrics logging
            if session_id in self.active_sessions:
                session_duration = time.time() - start_time
                session_data = self.active_sessions[session_id]

                logger.info(f"📊 Session {session_id} ended: "
                           f"duration={session_duration:.1f}s, "
                           f"requests={session_data['request_count']}, "
                           f"errors={session_data['error_count']}")

                del self.active_sessions[session_id]

            if session_id in session_metrics:
                del session_metrics[session_id]

    async def process_audio_chunk(self, session_id: str, base64_audio: str):
        """Enhanced audio chunk processing with production monitoring"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        processing_start = time.time()

        try:
            # Decode and enhance audio
            audio_bytes = base64.b64decode(base64_audio)
            enhanced_audio = self.audio_processor.process_audio_chunk(audio_bytes)

            websocket = session['websocket']
            session['request_count'] += 1

            # Enhanced VAD processing
            vad_result = self.vad.process_audio_chunk(enhanced_audio, session['assistant_speaking'])

            # Store VAD state for debugging
            session['vad_state'] = vad_result['debug_info']

            # Check for fast interruption
            if session['assistant_speaking'] and vad_result['debug_info'].get('speech_ratio', 0) > 0:
                volume_estimate = vad_result['debug_info'].get('speech_ratio', 0) * 1000
                if await self.interruption_handler.check_interruption(volume_estimate, True):
                    await self.handle_immediate_interruption(session_id)

            # Handle VAD events
            if vad_result['speech_started']:
                await self.handle_speech_started(session_id, enhanced_audio)
            elif vad_result['speech_ended']:
                await self.handle_speech_ended(session_id)
            elif vad_result['speech_detected']:
                await self.handle_ongoing_speech(session_id, enhanced_audio)

            # Track performance
            processing_time = time.time() - processing_start
            self.request_times.append(processing_time)

            # Update session metrics
            if session_id in session_metrics:
                session_metrics[session_id].total_audio_processed += len(audio_bytes)

        except Exception as e:
            logger.error(f"❌ Enhanced audio processing error: {e}")
            session['error_count'] += 1

            # Send error to client
            try:
                await session['websocket'].send_json({
                    "type": "error",
                    "message": "Audio processing failed",
                    "error_code": "AUDIO_PROCESSING_ERROR"
                })
            except:
                pass  # WebSocket might be closed

    async def handle_immediate_interruption(self, session_id: str):
        """Handle immediate interruption with minimal latency"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        logger.info(f"⚡ IMMEDIATE INTERRUPTION detected for session {session_id}")

        session['assistant_speaking'] = False
        session['interrupted'] = True

        # Track interruption
        if session_id in session_metrics:
            session_metrics[session_id].interruptions += 1

        websocket = session['websocket']

        # Send immediate stop signal
        await websocket.send_json({
            "type": "stop_audio",
            "priority": "immediate",
            "message": "Audio stopped"
        })

        await websocket.send_json({
            "type": "interruption",
            "message": "⚡ Quick interruption - ready for your input"
        })

    async def handle_speech_started(self, session_id: str, audio_bytes: bytes):
        """Enhanced speech start handling"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        logger.info("🗣️ Enhanced: Speech started")
        session['audio_buffer'] = bytearray()
        session['last_speech_time'] = time.time()

        # Handle interruption if needed
        if session['assistant_speaking']:
            await self.handle_immediate_interruption(session_id)

        await session['websocket'].send_json({
            "type": "vad_status",
            "status": "speaking",
            "confidence": session['vad_state'].get('speech_ratio', 0),
            "message": "👂 Listening with enhanced VAD..."
        })

        # Add to buffer
        session['audio_buffer'].extend(audio_bytes)

    async def handle_ongoing_speech(self, session_id: str, audio_bytes: bytes):
        """Handle ongoing speech with buffering"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        session['audio_buffer'].extend(audio_bytes)
        session['last_speech_time'] = time.time()

    async def handle_speech_ended(self, session_id: str):
        """Enhanced speech end handling"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        logger.info("✅ Enhanced: Speech ended, processing...")

        if len(session['audio_buffer']) > 0:
            # Optimize audio for Whisper before processing
            optimized_audio = self.audio_processor.optimize_for_whisper(bytes(session['audio_buffer']))
            await self.process_complete_speech(session_id, optimized_audio)
            session['audio_buffer'] = bytearray()

        await session['websocket'].send_json({
            "type": "vad_status",
            "status": "ready",
            "message": "🎤 Ready for next input"
        })

    async def process_complete_speech(self, session_id: str, speech_audio: bytes):
        """Enhanced speech processing with production error handling"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        if session.get('interrupted', False):
            logger.info("🛑 Skipping processing - interrupted")
            return

        websocket = session['websocket']
        request_start = time.time()

        try:
            logger.info(f"🎯 Processing {len(speech_audio)} bytes with enhanced pipeline")

            # 1. Transcription with enhanced error handling
            await websocket.send_json({
                "type": "processing_status",
                "status": "transcribing",
                "stage": "speech_to_text"
            })

            text = await self.transcribe_with_whisper(speech_audio)
            if not text:
                return

            await websocket.send_json({"type": "transcription", "text": text})

            # 2. LLM processing with context
            await websocket.send_json({
                "type": "processing_status",
                "status": "thinking",
                "stage": "generating_response"
            })

            response_text = await self.generate_response(session_id, text)
            if not response_text:
                return

            await websocket.send_json({"type": "response", "text": response_text})

            # 3. Enhanced TTS with optimization
            await websocket.send_json({
                "type": "processing_status",
                "status": "speaking",
                "stage": "text_to_speech"
            })

            session['assistant_speaking'] = True
            await self.generate_and_send_speech(session_id, response_text)

            # Update metrics
            request_time = time.time() - request_start
            if session_id in session_metrics:
                session_metrics[session_id].successful_completions += 1
                # Update rolling average
                current_avg = session_metrics[session_id].avg_response_time
                session_metrics[session_id].avg_response_time = (current_avg * 0.8) + (request_time * 0.2)

            logger.info(f"✅ Complete request processed in {request_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Enhanced speech processing error: {e}")
            session['error_count'] += 1

            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Processing failed - please try again",
                    "error_code": "PROCESSING_ERROR"
                })
            except:
                pass

        finally:
            if not session.get('interrupted', False):
                session['assistant_speaking'] = False
                await websocket.send_json({"type": "processing_status", "status": "ready"})

    async def transcribe_with_whisper(self, audio_bytes: bytes) -> Optional[str]:
        """Enhanced Whisper transcription with retries"""
        for attempt in range(3):  # 3 attempts
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                    data = aiohttp.FormData()
                    data.add_field('audio', audio_bytes, filename='recording.wav', content_type='audio/wav')

                    async with session.post(config.WHISPER_API, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = result.get('text', '').strip()
                            logger.info(f"📝 Whisper Success (attempt {attempt + 1}): {text}")
                            service_health.whisper_status = "healthy"
                            return text
                        else:
                            error_text = await response.text()
                            logger.warning(f"⚠️ Whisper attempt {attempt + 1} failed: {response.status}")

            except Exception as e:
                logger.warning(f"⚠️ Whisper attempt {attempt + 1} error: {e}")

            if attempt < 2:
                await asyncio.sleep(0.5)  # Brief delay before retry

        # All attempts failed
        service_health.whisper_status = "error"
        service_health.error_count += 1
        logger.error("❌ All Whisper attempts failed")
        return None

    async def generate_response(self, session_id: str, text: str) -> Optional[str]:
        """Enhanced LLM response generation"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        try:
            client = AsyncOpenAI(
                api_key=config.E2E_TOKEN,
                base_url=config.LLAMA_BASE_URL,
                timeout=20.0  # 20 second timeout
            )

            session['conversation_context'].append({"role": "user", "content": text})

            response = await client.chat.completions.create(
                model=config.LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Be conversational and concise. Provide natural, engaging responses."}
                ] + session['conversation_context'][-8:],  # Keep last 4 turns
                max_tokens=200,
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            session['conversation_context'].append({"role": "assistant", "content": response_text})

            logger.info(f"💬 LLM Response: {response_text}")
            service_health.llama_status = "healthy"
            return response_text

        except Exception as e:
            logger.error(f"❌ LLM generation error: {e}")
            service_health.llama_status = "error"
            service_health.error_count += 1
            return None

    async def generate_and_send_speech(self, session_id: str, text: str):
        """Enhanced TTS generation and sending"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as aio_session:
                glow_tts_url = "http://216.48.191.105:8000/v1/audio/speech"

                payload = {"input": text}
                headers = {"Content-Type": "application/json"}

                async with aio_session.post(glow_tts_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        json_response = await response.json()

                        if 'audio' in json_response and not session.get('interrupted', False):
                            audio_b64 = json_response['audio']

                            # Optimize TTS audio
                            audio_data = base64.b64decode(audio_b64)
                            enhanced_audio = self.audio_processor.optimize_for_tts_output(audio_data)
                            enhanced_b64 = base64.b64encode(enhanced_audio).decode()

                            await session['websocket'].send_json({
                                "type": "audio_response",
                                "data": enhanced_b64,
                                "content_type": "audio/wav",
                                "size": len(enhanced_audio),
                                "format": "enhanced_wav_22050_16bit_mono",
                                "enhanced": True
                            })

                            logger.info(f"🔊 Enhanced TTS sent: {len(enhanced_audio)} bytes")
                            service_health.tts_status = "healthy"

                        else:
                            if session.get('interrupted'):
                                logger.info("🛑 TTS skipped - user interrupted")
                            else:
                                logger.error("❌ No audio in TTS response")
                    else:
                        logger.error(f"❌ TTS API error: {response.status}")
                        service_health.tts_status = "error"

        except Exception as e:
            logger.error(f"❌ TTS generation error: {e}")
            service_health.tts_status = "error"
            service_health.error_count += 1

    async def process_legacy_audio(self, session_id: str, base64_audio: str):
        """Enhanced legacy audio support"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        logger.info("🔄 Processing legacy audio with enhanced pipeline")
        audio_bytes = base64.b64decode(base64_audio)
        enhanced_audio = self.audio_processor.optimize_for_whisper(audio_bytes)
        await self.process_complete_speech(session_id, enhanced_audio)

    async def get_service_status(self) -> dict:
        """Get comprehensive service status"""
        return {
            "services": {
                "whisper": service_health.whisper_status,
                "llama": service_health.llama_status,
                "tts": service_health.tts_status
            },
            "performance": {
                "avg_request_time_ms": np.mean(self.request_times) * 1000 if self.request_times else 0,
                "active_sessions": len(self.active_sessions),
                "total_errors": service_health.error_count,
                "total_requests": service_health.total_requests
            },
            "audio_processor": self.audio_processor.get_performance_stats(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
            }
        }

# Global handler and startup
start_time = time.time()
production_handler = ProductionVoiceHandler()

async def check_service_health():
    """Check health of external services"""
    service_health.last_check = time.time()

    # This could be expanded to actually ping services
    # For now, just reset status
    service_health.whisper_status = "unknown"
    service_health.llama_status = "unknown"
    service_health.tts_status = "unknown"

# Enhanced endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Production WebSocket endpoint"""
    await production_handler.handle_connection(websocket)

@app.get("/")
async def root():
    """Serve enhanced web client"""
    try:
        with open("web/index_enhanced.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return JSONResponse(
            content={"error": "Web interface not found"},
            status_code=404
        )

@app.get("/health")
async def health():
    """Comprehensive health check endpoint"""
    try:
        status = await production_handler.get_service_status()

        # Determine overall health
        services_healthy = all(
            s != "error" for s in [
                status["services"]["whisper"],
                status["services"]["llama"],
                status["services"]["tts"]
            ]
        )

        return JSONResponse(
            content={
                "status": "healthy" if services_healthy else "degraded",
                "service": "E2E Voice Assistant - Production",
                "version": "3.0.0",
                "timestamp": time.time(),
                **status
            },
            status_code=200 if services_healthy else 503
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            },
            status_code=500
        )

@app.get("/metrics")
async def metrics():
    """Detailed metrics endpoint for monitoring"""
    try:
        metrics_data = {
            "service_health": {
                "whisper_status": service_health.whisper_status,
                "llama_status": service_health.llama_status,
                "tts_status": service_health.tts_status,
                "total_errors": service_health.error_count,
                "total_requests": service_health.total_requests,
                "last_health_check": service_health.last_check
            },
            "session_metrics": {
                "active_sessions": len(production_handler.active_sessions),
                "total_sessions": len(session_metrics),
                "session_details": [
                    {
                        "session_id": m.session_id,
                        "duration": time.time() - m.start_time,
                        "audio_processed": m.total_audio_processed,
                        "completions": m.successful_completions,
                        "errors": m.errors,
                        "interruptions": m.interruptions,
                        "avg_response_time": m.avg_response_time
                    }
                    for m in session_metrics.values()
                ]
            },
            "performance": {
                "avg_request_time_ms": np.mean(production_handler.request_times) * 1000 if production_handler.request_times else 0,
                "request_times_last_100": list(production_handler.request_times),
                "audio_processing_stats": production_handler.audio_processor.get_performance_stats()
            },
            "recent_errors": list(recent_errors)[-20:],  # Last 20 errors
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "uptime_seconds": time.time() - start_time
            }
        }

        return JSONResponse(content=metrics_data)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Metrics collection failed: {str(e)}"},
            status_code=500
        )

@app.get("/debug/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint for specific session"""
    if session_id in production_handler.active_sessions:
        session = production_handler.active_sessions[session_id]
        return {
            "session_id": session_id,
            "active": True,
            "vad_state": session.get('vad_state', {}),
            "audio_stats": session.get('audio_stats', {}),
            "conversation_length": len(session.get('conversation_context', [])),
            "request_count": session.get('request_count', 0),
            "error_count": session.get('error_count', 0),
            "last_activity": session.get('last_activity', 0)
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    import numpy as np

    logger.info("🚀 PRODUCTION E2E VOICE ASSISTANT")
    logger.info("=" * 60)
    logger.info("✅ Enhanced VAD with Adaptive Thresholds")
    logger.info("✅ Advanced Audio Processing & Enhancement")
    logger.info("✅ Ultra-Fast Interruption Handling")
    logger.info("✅ Production Error Handling & Recovery")
    logger.info("✅ Comprehensive Monitoring & Metrics")
    logger.info("✅ Service Health Checks")
    logger.info("✅ Session Tracking & Analytics")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )