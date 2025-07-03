"""
Audio Waveform Generator for visualizing audio levels
"""

import numpy as np
from collections import deque
from typing import Optional
import logging
from .. import config

logger = logging.getLogger(__name__)


class AudioWaveformGenerator:
    """Generates waveform visualization from audio data"""

    def __init__(self, width=200, height=80, bars=16):
        self.width = width
        self.height = height
        self.bars = bars
        self.bar_width = width // bars
        self.bar_spacing = 1  # Pixels between bars
        self.audio_history = deque(maxlen=bars)
        self.smoothing_factor = 0.5  # Reduced from 0.85 for more responsive animation
        self.current_levels = np.zeros(bars)
        self.target_levels = np.zeros(bars)
        self.silence_threshold = 0.0005  # Lower threshold
        self.level_multiplier = 3.0  # Increased for better visibility
        self.decay_rate = 0.92  # Slightly faster decay for more responsive fade
        self.min_bar_height = 0.02  # Minimum visible bar height
        self.enable_reflection = False  # Set to True if you want reflections

        # Pre-calculate bar positions for efficiency
        self.bar_positions = []
        for i in range(bars):
            x_start = i * self.bar_width + self.bar_spacing
            x_end = (i + 1) * self.bar_width - self.bar_spacing
            self.bar_positions.append((x_start, x_end))

        # Initialize with zeros
        for _ in range(bars):
            self.audio_history.append(0.0)

    def process_audio_chunk(self, audio_data: bytes) -> np.ndarray:
        """Process audio chunk and return visualization levels"""
        if len(audio_data) < 2:
            # Apply slower decay when no audio
            self.current_levels = self.current_levels * self.decay_rate
            return self.current_levels

        # Convert bytes to numpy array with proper normalization
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Normalize to -1.0 to 1.0 with proper scaling
        audio_array = audio_array / 32767.0
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Calculate RMS
        rms = np.sqrt(np.mean(audio_array**2))

        # Only decay if truly silent
        if rms < self.silence_threshold:
            self.current_levels = self.current_levels * self.decay_rate
            return self.current_levels

        # Perform FFT for frequency analysis
        if len(audio_array) > self.bars * 2:
            # Apply Hann window
            window = np.hanning(len(audio_array))
            windowed = audio_array * window

            # Compute FFT with zero padding for better frequency resolution
            fft_size = 2048  # Fixed FFT size
            if len(windowed) < fft_size:
                windowed = np.pad(windowed, (0, fft_size - len(windowed)), "constant")
            else:
                windowed = windowed[:fft_size]

            fft = np.abs(np.fft.rfft(windowed))

            # Calculate sample rate from config
            sample_rate = config.AUDIO_SAMPLE_RATE
            freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)

            # Create logarithmic bins with better distribution
            min_freq = 40  # Start from 40Hz
            max_freq = 8000  # Up to 8kHz for better coverage
            log_freqs = np.logspace(
                np.log10(min_freq), np.log10(max_freq), self.bars + 1
            )

            bar_levels = np.zeros(self.bars)
            for i in range(self.bars):
                freq_mask = (freqs >= log_freqs[i]) & (freqs < log_freqs[i + 1])
                if np.any(freq_mask):
                    # Use maximum instead of mean for more responsive visualization
                    bar_levels[i] = np.max(fft[freq_mask])

            # Normalize with better scaling
            bar_levels = np.log1p(bar_levels * 100) / 8.0

            # Apply RMS envelope
            bar_levels = bar_levels * (0.3 + rms * 3.0)

            # Apply level multiplier
            bar_levels = bar_levels * self.level_multiplier

            # Enhanced bass response
            for i in range(min(10, self.bars)):
                bar_levels[i] *= 1.5 - i * 0.05

            # Apply minimum bar height for active audio
            bar_levels = np.maximum(bar_levels, self.min_bar_height)

            # Clamp to 0-1 range
            bar_levels = np.clip(bar_levels, 0, 1.0)
        else:
            # Fallback for very short chunks
            bar_levels = np.full(self.bars, rms * self.level_multiplier)

        # Smoother interpolation with faster response
        self.target_levels = bar_levels
        diff = self.target_levels - self.current_levels
        # Use 1 - smoothing_factor for the blend amount (more intuitive)
        # Lower smoothing_factor = faster response
        self.current_levels = self.current_levels + diff * (1 - self.smoothing_factor)

        # Apply a minimum change threshold to prevent stuck animation
        min_change = 0.001
        for i in range(len(self.current_levels)):
            if abs(diff[i]) > min_change:
                # Force update if difference is significant but animation is stuck
                if abs(self.current_levels[i] - self.target_levels[i]) > 0.1:
                    self.current_levels[i] = (
                        self.target_levels[i] * 0.8 + self.current_levels[i] * 0.2
                    )

        return self.current_levels

    def create_waveform_overlay(self, levels: np.ndarray) -> bytes:
        """Create RGBA overlay image data for waveform with synchronized bars"""
        # Create RGBA image buffer - using numpy for efficiency
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Fill with opaque background (original color)
        overlay[:, :] = [145, 156, 211, 255]  # R, G, B, A (255 = fully opaque)

        # Get mouth openness (single value that controls all bars)
        if len(levels) == 0:
            return overlay.tobytes()

        mouth_openness = levels[0] if len(levels) > 0 else 0.0

        # Only draw bars if there's some openness
        if mouth_openness < 0.005:
            return overlay.tobytes()  # Return background only when closed

        # Center line position
        y_center = self.height // 2

        # Create wave effect across all bars - mountain-like shape with randomness
        for i in range(self.bars):
            # Get pre-calculated bar position
            x_start, x_end = self.bar_positions[i]

            # Create mountain-like distribution (bell curve centered on middle bars)
            center_bar = self.bars / 2
            distance_from_center = abs(i - center_bar) / (self.bars / 2)

            # Mountain shape - higher in center, lower at edges
            mountain_factor = np.cos(distance_from_center * np.pi / 2) ** 1.5

            # Add random variation to each bar (but not too much to keep mountain shape)
            if not hasattr(self, "_bar_random_seeds"):
                self._bar_random_seeds = np.random.random(self.bars) * 2 - 1  # -1 to 1
                self._bar_random_time = 0

            # Slowly evolving random factor for each bar
            self._bar_random_time += 0.02
            random_variation = (
                self._bar_random_seeds[i]
                * np.sin(self._bar_random_time + i * 0.3)
                * 0.25
            )

            # Combine mouth openness, mountain shape, and random variation
            bar_level = mouth_openness * mountain_factor * (0.8 + random_variation)
            bar_level = max(0.05, min(1.0, bar_level))  # Keep in bounds

            # Calculate bar height
            bar_height = max(int(bar_level * self.height * 0.8), 2)

            # Create symmetrical bars (mirror effect)
            y_top = max(0, y_center - bar_height // 2)
            y_bottom = min(self.height, y_center + bar_height // 2)

            # Draw main bar with gradient effect
            for y in range(y_top, y_bottom):
                # Calculate distance from center for gradient
                distance_from_center = abs(y - y_center)
                if bar_height > 0:
                    gradient = 1.0 - (distance_from_center / (bar_height / 2 + 1))
                    gradient = max(0.3, min(1.0, gradient))
                else:
                    gradient = 1.0

                # Apply level-based intensity
                intensity = gradient * (0.7 + 0.3 * bar_level)

                # Set color with gradient (original green theme)
                color_r = int(17 * intensity)
                color_g = int(20 * intensity)
                color_b = int(35 * intensity)
                alpha = 255  # Always fully opaque

                # Draw the bar line
                overlay[y, x_start:x_end] = [color_r, color_g, color_b, alpha]

            # Add glow effect for high levels
            if bar_level > 0.5:
                glow_intensity = (bar_level - 0.5) * 2  # 0-1 range for levels 0.5-1

                # Top glow
                for glow_y in range(max(0, y_top - 2), y_top):
                    glow_alpha = 255  # Keep opaque
                    glow_factor = glow_intensity * (1 - (y_top - glow_y) / 2)
                    if glow_factor > 0:
                        # Blend glow with background
                        current_color = overlay[glow_y, x_start]
                        new_r = int(
                            current_color[0]
                            + (255 - current_color[0]) * glow_factor * 0.3
                        )
                        new_g = int(
                            current_color[1]
                            + (255 - current_color[1]) * glow_factor * 0.3
                        )
                        new_b = int(
                            current_color[2]
                            + (255 - current_color[2]) * glow_factor * 0.3
                        )
                        overlay[glow_y, x_start:x_end] = [
                            new_r,
                            new_g,
                            new_b,
                            glow_alpha,
                        ]

                # Bottom glow
                for glow_y in range(y_bottom, min(self.height, y_bottom + 2)):
                    glow_alpha = 255  # Keep opaque
                    glow_factor = glow_intensity * (1 - (glow_y - y_bottom) / 2)
                    if glow_factor > 0:
                        # Blend glow with background
                        current_color = overlay[glow_y, x_start]
                        new_r = int(
                            current_color[0]
                            + (255 - current_color[0]) * glow_factor * 0.3
                        )
                        new_g = int(
                            current_color[1]
                            + (255 - current_color[1]) * glow_factor * 0.3
                        )
                        new_b = int(
                            current_color[2]
                            + (255 - current_color[2]) * glow_factor * 0.3
                        )
                        overlay[glow_y, x_start:x_end] = [
                            new_r,
                            new_g,
                            new_b,
                            glow_alpha,
                        ]

        return overlay.tobytes()
