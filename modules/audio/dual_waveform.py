"""
Dual Waveform Generator for two robot streams with positioned overlays
"""

import numpy as np
import cv2
import time
import threading
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class DualWaveformGenerator:
    """Generates two separate waveforms for left and right robots"""

    def __init__(self, width: int = 200, height: int = 80, bars: int = 16):
        self.width = width
        self.height = height
        self.bars = bars
        
        # Waveform state for each robot
        self.left_levels = np.zeros(bars)
        self.right_levels = np.zeros(bars)
        self.left_target_levels = np.zeros(bars)
        self.right_target_levels = np.zeros(bars)
        
        # Animation parameters
        self.decay_factor = 0.85
        self.sensitivity = 1.5
        self.animation_lock = threading.Lock()
        
        # Visual parameters
        self.bar_width = max(1, (width - (bars - 1)) // bars)
        self.bar_spacing = 1
        
        # Colors for each robot
        self.left_color = (50, 205, 50)    # Lime green for left robot
        self.right_color = (30, 144, 255)  # Dodger blue for right robot
        self.glow_color = (255, 255, 255)
        
        # Speaking state
        self.left_speaking = False
        self.right_speaking = False
        
        logger.info(f"[DUAL_WAVEFORM] Initialized: {width}x{height}, {bars} bars, bar_width: {self.bar_width}")

    def update_left_levels(self, audio_data: bytes):
        """Update waveform levels for left robot"""
        self._update_robot_levels(audio_data, 'left')

    def update_right_levels(self, audio_data: bytes):
        """Update waveform levels for right robot"""
        self._update_robot_levels(audio_data, 'right')

    def _update_robot_levels(self, audio_data: bytes, robot: str):
        """Update waveform levels for a specific robot"""
        if not audio_data or len(audio_data) < 2:
            return

        try:
            # Convert audio to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS for speaking detection
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            is_speaking = rms > 100  # Threshold for speaking detection
            
            with self.animation_lock:
                if robot == 'left':
                    self.left_speaking = is_speaking
                    target_levels = self.left_target_levels
                else:
                    self.right_speaking = is_speaking
                    target_levels = self.right_target_levels

                if is_speaking:
                    # Generate waveform levels from audio
                    chunk_size = len(audio_array) // self.bars
                    if chunk_size > 0:
                        for i in range(self.bars):
                            start_idx = i * chunk_size
                            end_idx = min(start_idx + chunk_size, len(audio_array))
                            chunk = audio_array[start_idx:end_idx]
                            
                            # Calculate RMS for this frequency band
                            chunk_rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                            normalized_level = min(1.0, chunk_rms / 8000.0 * self.sensitivity)
                            target_levels[i] = normalized_level
                else:
                    # Decay to silence
                    target_levels *= 0.3

        except Exception as e:
            logger.error(f"[DUAL_WAVEFORM] Error updating {robot} levels: {e}")

    def animate_waveforms(self):
        """Animate both waveforms with smooth transitions"""
        with self.animation_lock:
            # Animate left robot waveform
            self.left_levels = (self.left_levels * self.decay_factor + 
                              self.left_target_levels * (1 - self.decay_factor))
            
            # Animate right robot waveform
            self.right_levels = (self.right_levels * self.decay_factor + 
                               self.right_target_levels * (1 - self.decay_factor))

    def generate_left_waveform(self) -> np.ndarray:
        """Generate waveform image for left robot"""
        return self._generate_robot_waveform('left')

    def generate_right_waveform(self) -> np.ndarray:
        """Generate waveform image for right robot"""
        return self._generate_robot_waveform('right')

    def _generate_robot_waveform(self, robot: str) -> np.ndarray:
        """Generate waveform image for a specific robot"""
        # Create transparent background (BGRA format)
        frame = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        with self.animation_lock:
            if robot == 'left':
                levels = self.left_levels.copy()
                speaking = self.left_speaking
                color = self.left_color
            else:
                levels = self.right_levels.copy()
                speaking = self.right_speaking
                color = self.right_color

        if not speaking:
            return frame  # Return transparent frame when not speaking

        # Draw waveform bars
        for i, level in enumerate(levels):
            x = i * (self.bar_width + self.bar_spacing)
            bar_height = int(level * self.height * 0.8)  # 80% of available height
            
            if bar_height > 0:
                y_start = (self.height - bar_height) // 2
                y_end = y_start + bar_height
                
                # Draw main bar
                cv2.rectangle(frame, 
                            (x, y_start), 
                            (x + self.bar_width - 1, y_end),
                            (*color, 255),  # Full opacity
                            -1)
                
                # Add glow effect for higher levels
                if level > 0.6:
                    glow_intensity = int((level - 0.6) * 255 / 0.4)
                    glow_color_with_alpha = (*self.glow_color, glow_intensity)
                    
                    # Outer glow
                    cv2.rectangle(frame,
                                (max(0, x - 1), max(0, y_start - 1)),
                                (min(self.width - 1, x + self.bar_width), min(self.height - 1, y_end + 1)),
                                glow_color_with_alpha,
                                1)

        return frame

    def get_combined_waveform_overlay(self, video_frame: np.ndarray, 
                                    left_position: Tuple[int, int], 
                                    right_position: Tuple[int, int]) -> np.ndarray:
        """Generate combined overlay with both waveforms positioned correctly"""
        # Create overlay frame matching video dimensions
        overlay = np.zeros((video_frame.shape[0], video_frame.shape[1], 4), dtype=np.uint8)
        
        # Generate individual waveforms
        left_waveform = self.generate_left_waveform()
        right_waveform = self.generate_right_waveform()
        
        # Position left waveform
        left_x, left_y = left_position
        if (left_x + self.width <= overlay.shape[1] and 
            left_y + self.height <= overlay.shape[0]):
            overlay[left_y:left_y + self.height, 
                   left_x:left_x + self.width] = left_waveform
        
        # Position right waveform
        right_x, right_y = right_position
        if (right_x + self.width <= overlay.shape[1] and 
            right_y + self.height <= overlay.shape[0]):
            overlay[right_y:right_y + self.height, 
                   right_x:right_x + self.width] = right_waveform
        
        return overlay

    def get_speaking_status(self) -> Tuple[bool, bool]:
        """Get speaking status for both robots (left, right)"""
        with self.animation_lock:
            return (self.left_speaking, self.right_speaking)

    def set_robot_speaking(self, robot: str, speaking: bool):
        """Manually set speaking status for a robot"""
        with self.animation_lock:
            if robot == 'left':
                self.left_speaking = speaking
            else:
                self.right_speaking = speaking

    def clear_waveforms(self):
        """Clear both waveforms"""
        with self.animation_lock:
            self.left_levels.fill(0)
            self.right_levels.fill(0)
            self.left_target_levels.fill(0)
            self.right_target_levels.fill(0)
            self.left_speaking = False
            self.right_speaking = False
            logger.info("[DUAL_WAVEFORM] All waveforms cleared")
