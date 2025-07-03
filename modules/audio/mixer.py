"""
Simple Audio Mixer for low latency streaming
"""

import numpy as np
import threading
import time
import logging
from collections import deque
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SimpleAudioMixer:
    """Simple audio mixer for low latency without dropping audio"""

    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = 2  # 16-bit
        self.bytes_per_frame = self.channels * self.bytes_per_sample

        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.total_bytes_buffered = 0

        # Pre-generate silence
        self.silence_chunk = bytes(
            int(sample_rate * 0.02 * self.bytes_per_frame)
        )  # 20ms silence
        
        # Add fade variables to prevent clicks
        self.fade_samples = int(sample_rate * 0.002)  # 2ms fade to prevent clicks
        self.last_output_level = 0.0  # Track last output level for smooth transitions

        # Volume multiplier - adjusted for cleaner audio
        self.volume_multiplier = 0.85  # Slightly higher but safer

        # Track if we're currently speaking
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_end_time = None  # Track when speech ended

        # Buffer health monitoring - improved thresholds for burst handling
        self.max_buffer_seconds = 8.0  # Increased from 5.0 to handle larger bursts
        self.last_buffer_report = time.time()
        self.min_buffer_target = 0.2  # Minimum 200ms for smooth playback
        
        # Burst handling
        self.max_safe_burst_buffer = 3.0  # Start draining faster after 3s
        self.emergency_clear_threshold = 15.0  # Increased from 5.0 - allow bigger buffers before emergency clear

        # Improved noise gate with hysteresis
        self.noise_gate_low = 30  # Lower threshold to turn on
        self.noise_gate_high = 50  # Higher threshold to turn off
        self.gate_active = False
        
        # Track incoming audio rate
        self.bytes_received_in_period = 0
        self.last_rate_report = time.time()
        self.total_bytes_received = 0
        self.last_incoming_rate = 0.0  # Track for external access
        
        # Silence warning throttling
        self.last_silence_warning = 0
        self.silence_warning_cooldown = 2.0  # Only warn every 2 seconds
        
        # Speech end callback for faster robot switching
        self.speech_end_callback = None
        self.speech_end_delay = 0.01  # ULTRA-INSTANT switching - 10ms delay only

    def set_speech_end_callback(self, callback):
        """Set callback to be called when speech ends"""
        self.speech_end_callback = callback

    def add_audio(self, audio_data: bytes):
        """Add audio with volume boost and improved noise reduction"""
        if not audio_data:
            return

        # Track incoming bytes
        self.bytes_received_in_period += len(audio_data)
        self.total_bytes_received += len(audio_data)
        
        # Report incoming rate every second
        current_time = time.time()
        time_since_report = current_time - self.last_rate_report
        if time_since_report >= 1.0:
            bytes_per_second = self.bytes_received_in_period / time_since_report
            audio_seconds_per_real_second = bytes_per_second / (self.sample_rate * self.bytes_per_frame)
            self.last_incoming_rate = audio_seconds_per_real_second
            if self.bytes_received_in_period > 0:
                logger.info(
                    f"[MIXER] Incoming rate: {bytes_per_second:.0f} bytes/sec = {audio_seconds_per_real_second:.1f}x realtime"
                )
            self.bytes_received_in_period = 0
            self.last_rate_report = current_time

        # Log incoming audio - removed to reduce spam
        # logger.debug(f"[MIXER] Received {len(audio_data)} bytes of audio from OpenAI")

        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Calculate RMS for the chunk
        rms = np.sqrt(np.mean(audio_array**2))
        # logger.debug(f"[MIXER] Audio RMS: {rms:.4f}")  # Removed to reduce spam

        # Improved noise gate with hysteresis
        if self.gate_active:
            # Gate is currently active (blocking noise)
            if rms > self.noise_gate_high:
                self.gate_active = False  # Open the gate
                # logger.debug(f"[MIXER] Noise gate opened (RMS {rms:.0f} > {self.noise_gate_high})")  # Removed spam
            else:
                # Still below threshold, apply gentle reduction
                audio_array = audio_array * 0.2
        else:
            # Gate is currently open (passing audio)
            if rms < self.noise_gate_low:
                self.gate_active = True  # Close the gate
                audio_array = audio_array * 0.2
                # logger.debug(f"[MIXER] Noise gate closed (RMS {rms:.0f} < {self.noise_gate_low})")  # Removed spam

        # Apply volume multiplier without aggressive clipping
        amplified = audio_array * self.volume_multiplier

        # Simple hard clipping instead of soft clipping
        amplified = np.clip(amplified, -32767, 32767)
        amplified_bytes = amplified.astype(np.int16).tobytes()

        with self.buffer_lock:
            self.audio_buffer.append(amplified_bytes)
            self.total_bytes_buffered += len(amplified_bytes)

            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = time.time()
                logger.info(
                    f"[MIXER] Speech started, buffer: {self.total_bytes_buffered} bytes"
                )

            # Reset speech end time when new audio arrives
            self.speech_end_time = None

            # Report buffer health periodically
            if time.time() - self.last_buffer_report > 1.0:
                buffer_seconds = self.total_bytes_buffered / (
                    self.sample_rate * self.bytes_per_frame
                )
                if buffer_seconds > 0.1:
                    logger.info(
                        f"[MIXER] Audio buffer: {buffer_seconds:.2f}s ({self.total_bytes_buffered} bytes, {len(self.audio_buffer)} chunks)"
                    )
                self.last_buffer_report = time.time()

    def get_audio_chunk(self, size: int) -> bytes:
        """Get audio chunk with burst-aware buffering"""
        # Ensure size is aligned to 2-byte boundaries for int16 samples
        aligned_size = (size // 2) * 2  # Round down to nearest even number
        if aligned_size == 0:
            aligned_size = 2  # Minimum one sample
        
        output = bytearray()

        with self.buffer_lock:
            # Log chunk request - removed to reduce spam
            # logger.debug(f"[MIXER] Requested chunk of {size} bytes (aligned to {aligned_size}), have {self.total_bytes_buffered} bytes buffered")
            
            # Aggressive low-latency reserve to minimize delays
            incoming_rate = getattr(self, 'last_incoming_rate', 1.0)
            
            if self.is_speaking:
                if incoming_rate > 4.0:
                    # ULTRA-AGGRESSIVE: Even extreme bursts get minimal reserve for instant response
                    min_reserve = aligned_size // 2  # Only 10ms reserve during bursts!
                elif incoming_rate > 2.0:
                    min_reserve = aligned_size // 2  # 10ms reserve
                else:
                    # For normal/slow audio, nearly zero reserve
                    min_reserve = aligned_size // 4  # Only 5ms reserve
            else:
                # Minimal reserve when not speaking
                min_reserve = 0  # No reserve - immediate response

            # Only get audio if we have more than the minimum reserve
            available_bytes = max(0, self.total_bytes_buffered - min_reserve)

            initial_available = available_bytes
            chunks_used = 0

            while len(output) < aligned_size and self.audio_buffer and available_bytes > 0:
                chunk = self.audio_buffer[0]  # Peek at first chunk
                needed = aligned_size - len(output)

                # Determine how much we can safely take
                can_take = min(len(chunk), needed, available_bytes)
                
                # Ensure can_take is also aligned to 2-byte boundaries
                can_take = (can_take // 2) * 2

                if can_take <= 0:
                    break

                if can_take == len(chunk):
                    # Take the whole chunk
                    self.audio_buffer.popleft()
                    output.extend(chunk)
                    self.total_bytes_buffered -= len(chunk)
                    available_bytes -= len(chunk)
                    chunks_used += 1
                else:
                    # Take partial chunk
                    chunk = self.audio_buffer.popleft()
                    output.extend(chunk[:can_take])
                    # Put remainder back
                    self.audio_buffer.appendleft(chunk[can_take:])
                    self.total_bytes_buffered -= can_take
                    available_bytes -= can_take
                    chunks_used += 0.5  # Partial chunk

            # Check if speech ended
            if self.is_speaking and self.total_bytes_buffered <= min_reserve:
                if not self.speech_end_time:
                    self.speech_end_time = time.time()
                # Only mark speech as ended after a delay
                elif time.time() - self.speech_end_time > self.speech_end_delay:  # Configurable delay
                    self.is_speaking = False
                    if self.speech_start_time:
                        duration = self.speech_end_time - self.speech_start_time
                        logger.info(f"[MIXER] Speech ended, duration: {duration:.2f}s")
                        
                        # Trigger speech end callback for faster robot switching
                        if self.speech_end_callback:
                            try:
                                self.speech_end_callback()
                            except Exception as e:
                                logger.error(f"[MIXER] Speech end callback error: {e}")

        # Fill with silence if needed - ensure alignment - OPTIMIZED FOR LOW LATENCY
        if len(output) < aligned_size:
            silence_needed = aligned_size - len(output)
            # Ensure silence_needed is also aligned
            silence_needed = (silence_needed // 2) * 2
            
            # MORE AGGRESSIVE: Only add silence if really necessary
            current_time = time.time()
            if self.is_speaking:
                # During speech, only add minimal silence and warn less frequently
                if silence_needed > aligned_size * 0.5:  # Only warn for larger gaps
                    if (incoming_rate < 3.0 and  # Increased threshold 
                        current_time - self.last_silence_warning > self.silence_warning_cooldown):
                        logger.warning(
                            f"[MIXER] Adding {silence_needed} bytes of silence during speech (buffer low)"
                        )
                        self.last_silence_warning = current_time
            
            # Always fill to prevent audio glitches, but with minimal impact
            if silence_needed > 0:
                output.extend(self.silence_chunk[:silence_needed])
        else:
            # logger.debug(f"[MIXER] Returned {len(output)} bytes from {chunks_used} chunks")  # Removed spam
            pass

        # Final safety check - ensure output is exactly the aligned size
        final_output = bytes(output[:aligned_size])
        
        # Apply anti-click processing to prevent audio pops
        if len(final_output) >= 4:  # Need at least 2 samples for processing
            final_output = self._apply_anti_click_processing(final_output)
        
        # Verify alignment
        if len(final_output) % 2 != 0:
            logger.error(f"[MIXER] Buffer alignment error: {len(final_output)} bytes is not aligned to 2-byte boundary")
            # Fix by truncating to aligned size
            final_output = final_output[:(len(final_output) // 2) * 2]
        
        return final_output

    def _apply_anti_click_processing(self, audio_data: bytes) -> bytes:
        """Apply anti-click processing to prevent audio pops"""
        if len(audio_data) < 4:
            return audio_data
        
        # Convert to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Check if this chunk starts or ends with significant level changes
        first_sample = audio_array[0]
        last_sample = audio_array[-1]
        
        # Apply small fade-in if starting from silence
        if abs(first_sample) > 100 and hasattr(self, 'last_output_level') and abs(self.last_output_level) < 50:
            fade_samples = min(48, len(audio_array) // 4)  # 2ms fade at 24kHz
            if fade_samples > 0:
                fade_in = np.linspace(0.0, 1.0, fade_samples)
                audio_array[:fade_samples] *= fade_in
        
        # Apply small fade-out if ending to silence
        if abs(last_sample) > 100:
            fade_samples = min(48, len(audio_array) // 4)  # 2ms fade at 24kHz  
            if fade_samples > 0:
                fade_out = np.linspace(1.0, 0.8, fade_samples)  # Don't fade to zero, just reduce
                audio_array[-fade_samples:] *= fade_out
        
        # Update last output level for next chunk
        self.last_output_level = float(last_sample)
        
        # Convert back to bytes
        return audio_array.astype(np.int16).tobytes()

    def clear_buffer(self):
        """Clear buffer for new response - with fade out"""
        with self.buffer_lock:
            cleared = self.total_bytes_buffered

            # If we're clearing during speech, add a small fade-out
            if self.is_speaking and self.audio_buffer:
                # Get a small chunk of the current audio for fade-out
                fade_samples = min(480, self.total_bytes_buffered // 2)  # 10ms at 24kHz
                if fade_samples > 0 and self.audio_buffer:
                    # Create fade-out
                    fade_data = bytearray()
                    while len(fade_data) < fade_samples and self.audio_buffer:
                        chunk = self.audio_buffer.popleft()
                        fade_data.extend(
                            chunk[: min(len(chunk), fade_samples - len(fade_data))]
                        )

                    # Apply fade-out curve
                    if fade_data:
                        fade_array = np.frombuffer(fade_data, dtype=np.int16).astype(
                            np.float32
                        )
                        fade_length = len(fade_array)
                        fade_curve = np.linspace(1.0, 0.0, fade_length)
                        fade_array = fade_array * fade_curve
                        fade_bytes = fade_array.astype(np.int16).tobytes()

                        # Clear buffer and add fade-out
                        self.audio_buffer.clear()
                        self.audio_buffer.append(fade_bytes)
                        self.total_bytes_buffered = len(fade_bytes)
                        cleared = cleared - len(fade_bytes)
                else:
                    # Just clear if we can't create fade
                    self.audio_buffer.clear()
                    self.total_bytes_buffered = 0
            else:
                # Not speaking, just clear
                self.audio_buffer.clear()
                self.total_bytes_buffered = 0

            self.is_speaking = False
            self.speech_end_time = time.time()
            self.gate_active = False  # Reset noise gate

            if cleared > 0:
                logger.info(
                    f"Cleared {cleared} bytes from buffer (fade-out applied: {self.total_bytes_buffered} bytes remain)"
                )

    def get_buffer_health(self) -> Dict[str, Any]:
        """Get buffer health statistics"""
        with self.buffer_lock:
            buffer_seconds = self.total_bytes_buffered / (
                self.sample_rate * self.bytes_per_frame
            )

            # Determine health status
            if buffer_seconds < 0.1:
                status = "empty"
            elif buffer_seconds < 0.5:
                status = "healthy"
            elif buffer_seconds < 1.0:
                status = "filling"
            elif buffer_seconds < 2.0:
                status = "warning"
            else:
                status = "critical"

            return {
                "bytes": self.total_bytes_buffered,
                "seconds": buffer_seconds,
                "is_speaking": self.is_speaking,
                "chunks": len(self.audio_buffer),
                "status": status,
                "gate_active": self.gate_active,
            } 