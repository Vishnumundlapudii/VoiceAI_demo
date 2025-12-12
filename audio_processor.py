"""
Enhanced Audio Processing Module
Provides production-quality audio handling with optimized encoding,
noise reduction, and faster processing for real-time applications.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import io
import wave
import audioop
from typing import Optional, Tuple, Union
import asyncio
import time
from dataclasses import dataclass
from loguru import logger

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    # Input/Output formats
    input_sample_rate: int = 16000
    output_sample_rate: int = 22050  # Higher quality for TTS output
    input_channels: int = 1
    output_channels: int = 1
    bit_depth: int = 16

    # Processing parameters
    chunk_size_ms: int = 50  # Smaller chunks for faster response
    overlap_ms: int = 10     # Overlap for smoother processing

    # Noise reduction
    enable_noise_reduction: bool = True
    noise_gate_threshold: float = 0.01

    # Audio enhancement
    enable_agc: bool = True  # Automatic Gain Control
    agc_target_level: float = 0.5
    enable_compressor: bool = True

    # Buffering
    max_buffer_duration_ms: int = 10000  # 10 seconds max buffer

class AudioProcessor:
    """
    Production-ready audio processor with real-time capabilities
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

        # Processing state
        self.input_buffer = bytearray()
        self.processed_buffer = bytearray()
        self.noise_profile = None
        self.agc_level = 1.0

        # Performance tracking
        self.processing_times = []
        self.chunk_count = 0

        # Initialize filters
        self._init_filters()

        logger.info(f"Audio Processor initialized: {self.config.input_sample_rate}Hz → {self.config.output_sample_rate}Hz")

    def _init_filters(self):
        """Initialize audio filters for processing"""
        nyquist = self.config.input_sample_rate / 2

        # High-pass filter for noise reduction (remove low-freq rumble)
        hp_cutoff = 80  # Hz
        self.hp_b, self.hp_a = scipy.signal.butter(2, hp_cutoff / nyquist, btype='high')

        # Low-pass filter for anti-aliasing
        lp_cutoff = min(7000, nyquist * 0.8)  # Hz
        self.lp_b, self.lp_a = scipy.signal.butter(4, lp_cutoff / nyquist, btype='low')

        # Notch filter for 50/60Hz power line noise
        for freq in [50, 60]:  # European/American power line frequencies
            if freq < nyquist:
                notch_b, notch_a = scipy.signal.iirnotch(freq, 30, self.config.input_sample_rate)
                if not hasattr(self, 'notch_filters'):
                    self.notch_filters = []
                self.notch_filters.append((notch_b, notch_a))

    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction techniques"""
        if not self.config.enable_noise_reduction:
            return audio_data

        # Convert to float for processing
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.copy()

        # High-pass filtering
        filtered = scipy.signal.filtfilt(self.hp_b, self.hp_a, audio_float)

        # Notch filters for power line noise
        if hasattr(self, 'notch_filters'):
            for b, a in self.notch_filters:
                filtered = scipy.signal.filtfilt(b, a, filtered)

        # Simple noise gate
        gate_threshold = self.config.noise_gate_threshold
        filtered = np.where(np.abs(filtered) < gate_threshold,
                           filtered * 0.1,  # Reduce but don't completely silence
                           filtered)

        return filtered

    def apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply Automatic Gain Control for consistent levels"""
        if not self.config.enable_agc:
            return audio_data

        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms > 0:
            # Target level adjustment
            target_gain = self.config.agc_target_level / rms

            # Smooth gain changes to avoid artifacts
            max_gain_change = 1.5  # Limit sudden changes
            target_gain = np.clip(target_gain,
                                 self.agc_level / max_gain_change,
                                 self.agc_level * max_gain_change)

            # Update AGC level with smoothing
            self.agc_level = 0.9 * self.agc_level + 0.1 * target_gain

            # Apply gain
            adjusted = audio_data * self.agc_level

            # Prevent clipping
            adjusted = np.clip(adjusted, -0.95, 0.95)

            return adjusted

        return audio_data

    def apply_compressor(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression for better voice clarity"""
        if not self.config.enable_compressor:
            return audio_data

        # Simple soft-knee compressor
        threshold = 0.6
        ratio = 3.0

        # Calculate levels
        abs_audio = np.abs(audio_data)

        # Apply compression above threshold
        mask = abs_audio > threshold
        compression_factor = 1 + (ratio - 1) * ((abs_audio - threshold) / (1 - threshold))
        compression_factor = np.where(mask, compression_factor, 1.0)

        # Apply compression while preserving sign
        compressed = audio_data / np.maximum(compression_factor, 1.0)

        return compressed

    def process_audio_chunk(self, audio_bytes: bytes, optimize_for_tts: bool = False) -> bytes:
        """
        Process audio chunk with all enhancements

        Args:
            audio_bytes: Raw audio data
            optimize_for_tts: If True, optimize for TTS quality (higher sample rate)

        Returns:
            Processed audio bytes
        """
        start_time = time.time()

        try:
            # Convert bytes to numpy array
            if len(audio_bytes) == 0:
                return audio_bytes

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_np) == 0:
                return audio_bytes

            # Convert to float for processing
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Apply audio enhancements
            processed = self.apply_noise_reduction(audio_float)
            processed = self.apply_agc(processed)
            processed = self.apply_compressor(processed)

            # Low-pass filter to prevent aliasing
            processed = scipy.signal.filtfilt(self.lp_b, self.lp_a, processed)

            # Sample rate conversion if needed
            if optimize_for_tts and self.config.output_sample_rate != self.config.input_sample_rate:
                # Upsample for better TTS quality
                num_samples = int(len(processed) * self.config.output_sample_rate / self.config.input_sample_rate)
                processed = scipy.signal.resample(processed, num_samples)

            # Convert back to int16
            processed_int16 = (processed * 32767).astype(np.int16)

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            self.chunk_count += 1

            # Log performance occasionally
            if self.chunk_count % 50 == 0:
                avg_time = np.mean(self.processing_times) * 1000
                logger.debug(f"🎵 Audio processing: avg {avg_time:.1f}ms/chunk")

            return processed_int16.tobytes()

        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            return audio_bytes  # Return original on error

    def create_wav_buffer(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> bytes:
        """Create properly formatted WAV buffer with optimization"""
        if sample_rate is None:
            sample_rate = self.config.input_sample_rate

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.config.input_channels)
                wav_file.setsampwidth(self.config.bit_depth // 8)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)

            return wav_buffer.getvalue()

        except Exception as e:
            logger.error(f"❌ WAV creation error: {e}")
            # Fallback to simple WAV creation
            return self._create_simple_wav(audio_bytes, sample_rate)

    def _create_simple_wav(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        """Fallback simple WAV creation"""
        import struct

        # WAV header
        header = struct.pack('<4sI4s4sIHHIIHH4sI',
                           b'RIFF',
                           36 + len(audio_bytes),
                           b'WAVE',
                           b'fmt ',
                           16,  # PCM format chunk size
                           1,   # PCM format
                           self.config.input_channels,
                           sample_rate,
                           sample_rate * self.config.input_channels * (self.config.bit_depth // 8),
                           self.config.input_channels * (self.config.bit_depth // 8),
                           self.config.bit_depth,
                           b'data',
                           len(audio_bytes))

        return header + audio_bytes

    def optimize_for_whisper(self, audio_bytes: bytes) -> bytes:
        """Optimize audio specifically for Whisper ASR"""
        # Whisper works best with 16kHz, mono, 16-bit
        processed = self.process_audio_chunk(audio_bytes, optimize_for_tts=False)
        return self.create_wav_buffer(processed, self.config.input_sample_rate)

    def optimize_for_tts_output(self, audio_bytes: bytes) -> bytes:
        """Optimize audio for TTS output quality"""
        # Higher quality for TTS output
        processed = self.process_audio_chunk(audio_bytes, optimize_for_tts=True)
        return self.create_wav_buffer(processed, self.config.output_sample_rate)

    def get_performance_stats(self) -> dict:
        """Get audio processing performance statistics"""
        if not self.processing_times:
            return {"status": "no_data"}

        return {
            "avg_processing_time_ms": np.mean(self.processing_times) * 1000,
            "max_processing_time_ms": np.max(self.processing_times) * 1000,
            "total_chunks_processed": self.chunk_count,
            "current_agc_level": self.agc_level,
            "buffer_size_bytes": len(self.input_buffer)
        }

    def reset(self):
        """Reset processor state"""
        self.input_buffer.clear()
        self.processed_buffer.clear()
        self.noise_profile = None
        self.agc_level = 1.0
        self.processing_times.clear()
        self.chunk_count = 0
        logger.info("🔄 Audio processor state reset")

# Enhanced interruption handler for faster response
class InterruptionHandler:
    """
    Handles audio interruptions with minimal latency for natural conversations
    """

    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.interrupt_threshold = 500  # ms for very fast interruption
        self.last_interrupt_time = 0
        self.interrupt_cooldown = 0.5  # seconds

    async def check_interruption(self, volume_level: float, is_assistant_speaking: bool) -> bool:
        """Check if user is interrupting with optimized thresholds"""
        current_time = time.time()

        # Prevent rapid-fire interruptions
        if current_time - self.last_interrupt_time < self.interrupt_cooldown:
            return False

        if is_assistant_speaking:
            # Very sensitive during assistant speech
            interrupt_threshold = 600 * self.sensitivity

            if volume_level > interrupt_threshold:
                self.last_interrupt_time = current_time
                logger.info(f"⚡ FAST INTERRUPTION: volume={volume_level:.0f}, threshold={interrupt_threshold:.0f}")
                return True

        return False