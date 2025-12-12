"""
Enhanced VAD (Voice Activity Detection) Module
Provides production-ready voice detection with adaptive thresholds,
noise suppression, and improved accuracy.
"""

import numpy as np
import scipy.signal
from typing import Optional, Tuple
import time
from dataclasses import dataclass
from loguru import logger

@dataclass
class VADConfig:
    """Configuration for Enhanced VAD"""
    # Thresholds
    initial_energy_threshold: float = 0.01
    silence_threshold_multiplier: float = 2.0
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 500

    # Adaptive parameters
    adaptation_rate: float = 0.1
    noise_floor_update_rate: float = 0.05
    energy_history_size: int = 100

    # Audio processing
    preemphasis_coeff: float = 0.97
    frame_size_ms: int = 20
    hop_size_ms: int = 10

class EnhancedVAD:
    """
    Production-ready VAD with adaptive thresholds and noise suppression
    """

    def __init__(self, sample_rate: int = 16000, config: Optional[VADConfig] = None):
        self.sample_rate = sample_rate
        self.config = config or VADConfig()

        # Adaptive thresholds
        self.energy_threshold = self.config.initial_energy_threshold
        self.noise_floor = 0.0
        self.energy_history = []

        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_speech_time = None

        # Frame processing
        self.frame_size = int(self.config.frame_size_ms * self.sample_rate / 1000)
        self.hop_size = int(self.config.hop_size_ms * self.sample_rate / 1000)

        # High-pass filter for noise reduction
        nyquist = self.sample_rate / 2
        cutoff = 300  # Hz - remove low frequency noise
        self.b, self.a = scipy.signal.butter(4, cutoff / nyquist, btype='high')

        logger.info(f"Enhanced VAD initialized: sample_rate={sample_rate}, frame_size={self.frame_size}")

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve VAD accuracy"""
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # High-pass filter to remove low-frequency noise
        filtered = scipy.signal.filtfilt(self.b, self.a, audio_data)

        # Pre-emphasis filter
        emphasized = np.append(filtered[0], filtered[1:] - self.config.preemphasis_coeff * filtered[:-1])

        return emphasized

    def compute_energy_features(self, frame: np.ndarray) -> dict:
        """Compute multiple energy-based features for robust detection"""
        # Short-term energy
        energy = np.mean(frame ** 2)

        # Zero-crossing rate (indicates voiced vs unvoiced)
        zcr = np.mean(np.diff(np.sign(frame)) != 0)

        # Spectral centroid (frequency distribution)
        fft = np.fft.fft(frame)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(frame), 1/self.sample_rate)[:len(fft)//2]

        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0

        # High-frequency energy ratio
        high_freq_energy = np.sum(magnitude[len(magnitude)//2:])
        total_energy = np.sum(magnitude)
        hf_ratio = high_freq_energy / (total_energy + 1e-10)

        return {
            'energy': energy,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'hf_ratio': hf_ratio
        }

    def update_adaptive_threshold(self, current_energy: float, is_speech: bool):
        """Update adaptive energy threshold based on recent audio"""
        # Update energy history
        self.energy_history.append(current_energy)
        if len(self.energy_history) > self.config.energy_history_size:
            self.energy_history.pop(0)

        # Update noise floor (during non-speech periods)
        if not is_speech and len(self.energy_history) > 10:
            recent_energies = self.energy_history[-10:]
            estimated_noise = np.percentile(recent_energies, 20)  # 20th percentile as noise floor
            self.noise_floor = (1 - self.config.noise_floor_update_rate) * self.noise_floor + \
                              self.config.noise_floor_update_rate * estimated_noise

        # Adaptive threshold = noise_floor * multiplier
        self.energy_threshold = max(
            self.noise_floor * self.config.silence_threshold_multiplier,
            self.config.initial_energy_threshold
        )

    def detect_speech_in_frame(self, features: dict, assistant_speaking: bool = False) -> bool:
        """Detect speech in a single frame using multiple features"""
        energy = features['energy']
        zcr = features['zcr']
        spectral_centroid = features['spectral_centroid']
        hf_ratio = features['hf_ratio']

        # Primary energy-based detection
        energy_speech = energy > self.energy_threshold

        # Additional speech indicators
        # Higher ZCR often indicates speech
        zcr_speech = zcr > 0.1 and zcr < 0.8  # Too high ZCR might be noise

        # Speech typically has mid-range spectral centroid
        spectral_speech = 200 < spectral_centroid < 4000

        # Speech has balanced frequency distribution
        hf_speech = 0.1 < hf_ratio < 0.7

        # Combine features (energy is primary, others are supporting)
        speech_indicators = [energy_speech, zcr_speech, spectral_speech, hf_speech]
        speech_score = sum(speech_indicators)

        # During assistant speaking, be more sensitive (lower threshold for interruption)
        required_indicators = 2 if assistant_speaking else 3

        is_speech_detected = speech_score >= required_indicators

        # Update adaptive threshold
        self.update_adaptive_threshold(energy, is_speech_detected)

        return is_speech_detected

    def process_audio_chunk(self, audio_bytes: bytes, assistant_speaking: bool = False) -> dict:
        """
        Process audio chunk and return VAD decision with detailed info

        Returns:
            dict with keys: 'speech_detected', 'speech_started', 'speech_ended',
                          'confidence', 'debug_info'
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_np) == 0:
                return {'speech_detected': False, 'speech_started': False, 'speech_ended': False}

            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_np)

            # Process in frames
            speech_frames = 0
            total_frames = 0

            for i in range(0, len(processed_audio) - self.frame_size, self.hop_size):
                frame = processed_audio[i:i + self.frame_size]

                # Compute features
                features = self.compute_energy_features(frame)

                # Detect speech in frame
                frame_has_speech = self.detect_speech_in_frame(features, assistant_speaking)

                if frame_has_speech:
                    speech_frames += 1
                total_frames += 1

            # Overall speech decision for this chunk
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                chunk_has_speech = speech_ratio > 0.3  # 30% of frames must have speech
            else:
                chunk_has_speech = False

            # State management
            current_time = time.time()
            speech_started = False
            speech_ended = False

            if chunk_has_speech:
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    speech_started = True
                    logger.debug(f"🗣️ Speech started (confidence: {speech_ratio:.2f})")

                self.last_speech_time = current_time
                self.silence_start_time = None

            else:
                # No speech detected
                if self.is_speaking:
                    if self.silence_start_time is None:
                        self.silence_start_time = current_time

                    # Check if enough silence has passed
                    silence_duration = current_time - self.silence_start_time
                    required_silence = self.config.min_silence_duration_ms / 1000.0

                    if silence_duration >= required_silence:
                        # Speech ended
                        self.is_speaking = False
                        speech_ended = True
                        logger.debug(f"✅ Speech ended after {silence_duration:.2f}s silence")

            debug_info = {
                'speech_ratio': speech_ratio,
                'speech_frames': speech_frames,
                'total_frames': total_frames,
                'energy_threshold': self.energy_threshold,
                'noise_floor': self.noise_floor,
                'assistant_speaking': assistant_speaking
            }

            return {
                'speech_detected': self.is_speaking,
                'speech_started': speech_started,
                'speech_ended': speech_ended,
                'confidence': speech_ratio,
                'debug_info': debug_info
            }

        except Exception as e:
            logger.error(f"❌ Enhanced VAD error: {e}")
            return {
                'speech_detected': False,
                'speech_started': False,
                'speech_ended': False,
                'confidence': 0.0,
                'debug_info': {'error': str(e)}
            }

    def reset(self):
        """Reset VAD state (useful for new sessions)"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_speech_time = None
        self.energy_history.clear()
        logger.info("🔄 Enhanced VAD state reset")