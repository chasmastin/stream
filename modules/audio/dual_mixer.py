"""
Dual Audio Mixer for handling two robot audio streams
"""

import numpy as np
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DualAudioMixer:
    """Audio mixer that handles two separate robot audio streams"""

    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = 2  # 16-bit
        self.bytes_per_frame = self.channels * self.bytes_per_sample

        # Separate buffers for each robot
        self.left_buffer = deque()
        self.right_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # Track buffered bytes for each robot
        self.left_bytes_buffered = 0
        self.right_bytes_buffered = 0

        # Pre-generate silence
        self.silence_chunk = bytes(
            int(sample_rate * 0.02 * self.bytes_per_frame)
        )  # 20ms silence
        
        # Fade variables to prevent clicks
        self.fade_samples = int(sample_rate * 0.002)  # 2ms fade
        self.left_last_level = 0.0
        self.right_last_level = 0.0

        # Volume multiplier for each robot
        self.left_volume = 0.7  # Left robot volume
        self.right_volume = 0.7  # Right robot volume

        # Track speaking state for each robot
        self.left_speaking = False
        self.right_speaking = False
        self.left_speech_start = None
        self.right_speech_start = None

        # Buffer health monitoring
        self.max_buffer_seconds = 5.0
        self.last_buffer_report = time.time()

        # Noise gate settings
        self.noise_gate_low = 30
        self.noise_gate_high = 50
        self.left_gate_active = False
        self.right_gate_active = False
        
        # Track incoming rates
        self.left_bytes_received = 0
        self.right_bytes_received = 0
        self.last_rate_report = time.time()

    def add_left_audio(self, audio_data: bytes):
        """Add audio data for left robot"""
        self._add_robot_audio(audio_data, 'left')

    def add_right_audio(self, audio_data: bytes):
        """Add audio data for right robot"""
        self._add_robot_audio(audio_data, 'right')

    def _add_robot_audio(self, audio_data: bytes, robot: str):
        """Add audio data for a specific robot"""
        if not audio_data:
            return

        # Track incoming bytes
        if robot == 'left':
            self.left_bytes_received += len(audio_data)
        else:
            self.right_bytes_received += len(audio_data)

        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Calculate RMS for noise gate
        rms = np.sqrt(np.mean(audio_array**2))

        # Apply noise gate
        if robot == 'left':
            gate_active = self.left_gate_active
        else:
            gate_active = self.right_gate_active

        if gate_active:
            if rms > self.noise_gate_high:
                if robot == 'left':
                    self.left_gate_active = False
                else:
                    self.right_gate_active = False
            else:
                audio_array = audio_array * 0.2
        else:
            if rms < self.noise_gate_low:
                if robot == 'left':
                    self.left_gate_active = True
                else:
                    self.right_gate_active = True
                audio_array = audio_array * 0.2

        # Apply volume multiplier
        volume = self.left_volume if robot == 'left' else self.right_volume
        amplified = audio_array * volume

        # Clip to prevent distortion
        amplified = np.clip(amplified, -32767, 32767)
        amplified_bytes = amplified.astype(np.int16).tobytes()

        with self.buffer_lock:
            # Add to appropriate buffer
            if robot == 'left':
                self.left_buffer.append(amplified_bytes)
                self.left_bytes_buffered += len(amplified_bytes)
                
                if not self.left_speaking:
                    self.left_speaking = True
                    self.left_speech_start = time.time()
                    logger.info(f"[DUAL_MIXER] Left robot started speaking")
            else:
                self.right_buffer.append(amplified_bytes)
                self.right_bytes_buffered += len(amplified_bytes)
                
                if not self.right_speaking:
                    self.right_speaking = True
                    self.right_speech_start = time.time()
                    logger.info(f"[DUAL_MIXER] Right robot started speaking")

            # Report buffer health periodically
            self._report_buffer_health()

    def get_audio_chunk(self, size: int) -> bytes:
        """Get mixed audio chunk from both robots"""
        aligned_size = (size // 2) * 2  # Align to 2-byte boundaries
        if aligned_size == 0:
            aligned_size = 2

        left_audio = self._get_robot_audio_chunk('left', aligned_size)
        right_audio = self._get_robot_audio_chunk('right', aligned_size)

        # Mix the audio streams
        mixed_audio = self._mix_audio_streams(left_audio, right_audio)
        
        return mixed_audio

    def _get_robot_audio_chunk(self, robot: str, size: int) -> bytes:
        """Get audio chunk from a specific robot's buffer"""
        output = bytearray()
        
        if robot == 'left':
            buffer = self.left_buffer
            bytes_buffered = self.left_bytes_buffered
        else:
            buffer = self.right_buffer
            bytes_buffered = self.right_bytes_buffered

        with self.buffer_lock:
            # Get audio from buffer
            while len(output) < size and buffer:
                chunk = buffer.popleft()
                needed = size - len(output)
                
                if len(chunk) <= needed:
                    output.extend(chunk)
                    if robot == 'left':
                        self.left_bytes_buffered -= len(chunk)
                    else:
                        self.right_bytes_buffered -= len(chunk)
                else:
                    # Take partial chunk
                    output.extend(chunk[:needed])
                    remainder = chunk[needed:]
                    buffer.appendleft(remainder)
                    if robot == 'left':
                        self.left_bytes_buffered -= needed
                    else:
                        self.right_bytes_buffered -= needed

            # Pad with silence if needed
            if len(output) < size:
                silence_needed = size - len(output)
                output.extend(b'\x00' * silence_needed)

            # Update speaking status
            if robot == 'left' and self.left_speaking and not self.left_buffer:
                self.left_speaking = False
                logger.info(f"[DUAL_MIXER] Left robot stopped speaking")
            elif robot == 'right' and self.right_speaking and not self.right_buffer:
                self.right_speaking = False
                logger.info(f"[DUAL_MIXER] Right robot stopped speaking")

        return bytes(output)

    def _mix_audio_streams(self, left_audio: bytes, right_audio: bytes) -> bytes:
        """Mix two audio streams together"""
        if len(left_audio) != len(right_audio):
            # Pad shorter stream with silence
            max_len = max(len(left_audio), len(right_audio))
            left_audio = left_audio.ljust(max_len, b'\x00')
            right_audio = right_audio.ljust(max_len, b'\x00')

        # Convert to numpy arrays
        left_array = np.frombuffer(left_audio, dtype=np.int16).astype(np.float32)
        right_array = np.frombuffer(right_audio, dtype=np.int16).astype(np.float32)

        # Mix the streams - if both robots are speaking, prioritize the active speaker
        if self.left_speaking and self.right_speaking:
            # Both speaking - mix with reduced volumes to prevent clipping
            mixed = (left_array * 0.6) + (right_array * 0.6)
        elif self.left_speaking:
            # Only left speaking
            mixed = left_array + (right_array * 0.1)  # Add small amount of right for ambient
        elif self.right_speaking:
            # Only right speaking
            mixed = right_array + (left_array * 0.1)  # Add small amount of left for ambient
        else:
            # Neither speaking - mix equally
            mixed = (left_array + right_array) * 0.5

        # Clip to prevent distortion
        mixed = np.clip(mixed, -32767, 32767)
        
        return mixed.astype(np.int16).tobytes()

    def _report_buffer_health(self):
        """Report buffer health for both robots"""
        current_time = time.time()
        if current_time - self.last_buffer_report > 2.0:  # Report every 2 seconds
            left_seconds = self.left_bytes_buffered / (self.sample_rate * self.bytes_per_frame)
            right_seconds = self.right_bytes_buffered / (self.sample_rate * self.bytes_per_frame)
            
            if left_seconds > 0.1 or right_seconds > 0.1:
                logger.info(
                    f"[DUAL_MIXER] Buffer health - Left: {left_seconds:.2f}s, Right: {right_seconds:.2f}s"
                )
            
            self.last_buffer_report = current_time

    def get_robot_speaking_status(self) -> Tuple[bool, bool]:
        """Get speaking status for both robots (left, right)"""
        return (self.left_speaking, self.right_speaking)

    def clear_buffers(self):
        """Clear all audio buffers"""
        with self.buffer_lock:
            self.left_buffer.clear()
            self.right_buffer.clear()
            self.left_bytes_buffered = 0
            self.right_bytes_buffered = 0
            self.left_speaking = False
            self.right_speaking = False
            logger.info("[DUAL_MIXER] All buffers cleared")

    def get_buffer_health(self) -> Dict[str, Any]:
        """Get buffer health information"""
        with self.buffer_lock:
            left_seconds = self.left_bytes_buffered / (self.sample_rate * self.bytes_per_frame)
            right_seconds = self.right_bytes_buffered / (self.sample_rate * self.bytes_per_frame)
            
            return {
                "left_seconds": left_seconds,
                "right_seconds": right_seconds,
                "left_speaking": self.left_speaking,
                "right_speaking": self.right_speaking,
                "left_chunks": len(self.left_buffer),
                "right_chunks": len(self.right_buffer)
            }
