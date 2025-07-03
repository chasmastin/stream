#!/usr/bin/env python3
"""
Final working streamer with Ultra-Fast YouTube Chat Monitor and Audio Waveform
- YouTube chat monitoring with minimal latency
- Simulated chat for Twitch streams
- OpenAI Realtime API integration
- Ultra-low latency audio (FIXED - no skipping, no noise)
- Real video background support (FIXED - full screen)
- Audio waveform visualization overlay (FIXED - no freezing)
"""

import os
import sys
import json
import time
import queue
import threading
import logging
import subprocess
import random
import asyncio
import websockets
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from collections import deque

# YouTube API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# GStreamer imports
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib, GObject, GstVideo

# Initialize GStreamer FIRST
Gst.init(None)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
# YouTube Credentials
YOUTUBE_CLIENT_ID = (
    ""
)
YOUTUBE_CLIENT_SECRET = ""
YOUTUBE_REFRESH_TOKEN = ""

# Twitch Configuration - BOGOTÁ SERVER
TWITCH_RTMP_URL = "rtmp://bog01.contribute.live-video.net/app"

# OpenAI API Key
OPENAI_API_KEY = ""

# OpenAI Realtime API Configuration
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"
OPENAI_VOICE = "alloy"

# YouTube RTMP URL
YOUTUBE_RTMP_BASE_URL = "rtmp://a.rtmp.youtube.com/live2"

# Background video path
BACKGROUND_VIDEO_PATH = "output_fixed.mp4"

# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FRAMERATE = 30
VIDEO_BITRATE = 2500

# Audio settings
AUDIO_BITRATE = 128000
AUDIO_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1

# Waveform settings
WAVEFORM_WIDTH = 200  # Reduced from 800
WAVEFORM_HEIGHT = 80  # Reduced from 200
WAVEFORM_BARS = 16  # Reduced from 64 for better visibility at smaller size
WAVEFORM_COLOR = (17, 20, 35)  # Green color
WAVEFORM_GLOW_COLOR = (255, 255, 255)  # Lighter green for glow effect

# Simulated Twitch Chat Messages
TWITCH_SIMULATED_MESSAGES = [
    {
        "author": "TechGuru42",
        "message": "Hey AI! What's your favorite programming language?",
    },
    {"author": "StreamerFan99", "message": "This is so cool! 🔥"},
    {"author": "CodingNinja", "message": "Can you explain how neural networks work?"},
    {"author": "GameDev2024", "message": "What games do you play?"},
    {"author": "AIEnthusiast", "message": "Are you using GPT-4?"},
    {"author": "RandomViewer", "message": "First time here, this is amazing!"},
    {"author": "PythonMaster", "message": "Python > JavaScript 😎"},
    {"author": "TwitchMod", "message": "Keep up the great content!"},
    {"author": "CuriousCat", "message": "How long have you been streaming?"},
    {"author": "TechStudent", "message": "Can AI help me with homework?"},
    {"author": "CountingFan", "message": "Can you count from 1 to 10 slowly?"},
]

# YouTube scopes
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube",
]


class AudioWaveformGenerator:
    """Generates waveform visualization from audio data"""

    def __init__(
        self, width=WAVEFORM_WIDTH, height=WAVEFORM_HEIGHT, bars=WAVEFORM_BARS
    ):
        self.width = width
        self.height = height
        self.bars = bars
        self.bar_width = width // bars
        self.bar_spacing = 1  # Pixels between bars
        self.audio_history = deque(maxlen=bars)
        self.smoothing_factor = 0.85  # Increased for smoother transitions
        self.current_levels = np.zeros(bars)
        self.target_levels = np.zeros(bars)
        self.silence_threshold = 0.0005  # Lower threshold
        self.level_multiplier = 3.0  # Increased for better visibility
        self.decay_rate = 0.95  # Slower decay
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
            freqs = np.fft.rfftfreq(len(windowed), 1.0 / AUDIO_SAMPLE_RATE)

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

        # Smoother interpolation
        self.target_levels = bar_levels
        diff = self.target_levels - self.current_levels
        self.current_levels = self.current_levels + diff * (1 - self.smoothing_factor)

        return self.current_levels

    def create_waveform_overlay(self, levels: np.ndarray) -> bytes:
        """Create RGBA overlay image data for waveform with opaque background"""
        # Create RGBA image buffer - using numpy for efficiency
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Fill with opaque black background (or any color you want)
        # For black background:
        overlay[:, :] = [145, 156, 211, 255]  # R, G, B, A (255 = fully opaque)

        # Or for a dark gray background:
        # overlay[:, :] = [32, 32, 32, 255]

        # Or for a custom color background (e.g., dark blue):
        # overlay[:, :] = [10, 20, 40, 255]

        # Only draw if there's actual audio (not silence)
        max_level = np.max(levels)
        if max_level < 0.005:  # Lower threshold for drawing
            return overlay.tobytes()  # Return opaque background even when silent

        # Center line position
        y_center = self.height // 2

        # Draw waveform bars
        for i, level in enumerate(levels):
            if level < 0.005:  # Skip very low levels
                continue

            # Get pre-calculated bar position
            x_start, x_end = self.bar_positions[i]

            # Calculate bar height with minimum
            bar_height = max(int(level * self.height * 0.8), 2)

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
                intensity = gradient * (0.7 + 0.3 * level)

                # Set color with gradient (green theme)
                color_r = int(WAVEFORM_COLOR[0] * intensity)
                color_g = int(WAVEFORM_COLOR[1] * intensity)
                color_b = int(WAVEFORM_COLOR[2] * intensity)
                alpha = 255  # Always fully opaque

                # Draw the bar line
                overlay[y, x_start:x_end] = [color_r, color_g, color_b, alpha]

            # Add glow effect for high levels
            if level > 0.5:
                glow_intensity = (level - 0.5) * 2  # 0-1 range for levels 0.5-1

                # Top glow
                for glow_y in range(max(0, y_top - 2), y_top):
                    glow_alpha = 255  # Keep opaque
                    glow_factor = glow_intensity * (1 - (y_top - glow_y) / 2)
                    if glow_factor > 0:
                        # Blend glow with background
                        current_color = overlay[glow_y, x_start]
                        new_r = int(
                            current_color[0]
                            + (WAVEFORM_GLOW_COLOR[0] - current_color[0])
                            * glow_factor
                            * 0.3
                        )
                        new_g = int(
                            current_color[1]
                            + (WAVEFORM_GLOW_COLOR[1] - current_color[1])
                            * glow_factor
                            * 0.3
                        )
                        new_b = int(
                            current_color[2]
                            + (WAVEFORM_GLOW_COLOR[2] - current_color[2])
                            * glow_factor
                            * 0.3
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
                            + (WAVEFORM_GLOW_COLOR[0] - current_color[0])
                            * glow_factor
                            * 0.3
                        )
                        new_g = int(
                            current_color[1]
                            + (WAVEFORM_GLOW_COLOR[1] - current_color[1])
                            * glow_factor
                            * 0.3
                        )
                        new_b = int(
                            current_color[2]
                            + (WAVEFORM_GLOW_COLOR[2] - current_color[2])
                            * glow_factor
                            * 0.3
                        )
                        overlay[glow_y, x_start:x_end] = [
                            new_r,
                            new_g,
                            new_b,
                            glow_alpha,
                        ]

            # Add subtle reflection effect (optional, looks cool)
            if self.enable_reflection and bar_height > 4:
                reflection_height = bar_height // 4
                for ry in range(reflection_height):
                    y_pos = y_bottom + ry + 1
                    if y_pos < self.height:
                        reflection_factor = (1 - ry / reflection_height) * level * 0.3
                        if reflection_factor > 0:
                            # Blend reflection with background
                            current_color = overlay[y_pos, x_start]
                            new_r = int(
                                current_color[0]
                                + (color_r // 2 - current_color[0]) * reflection_factor
                            )
                            new_g = int(
                                current_color[1]
                                + (color_g // 2 - current_color[1]) * reflection_factor
                            )
                            new_b = int(
                                current_color[2]
                                + (color_b // 2 - current_color[2]) * reflection_factor
                            )
                            overlay[y_pos, x_start:x_end] = [new_r, new_g, new_b, 255]

        return overlay.tobytes()


class SimpleRealtimeAPIClient:
    """Simplified OpenAI Realtime API WebSocket Client with audio level callback"""

    def __init__(
        self, api_key: str, audio_callback: Callable, level_callback: Callable = None
    ):
        self.api_key = api_key
        self.ws = None
        self.audio_callback = audio_callback
        self.level_callback = level_callback
        self.is_connected = False
        self.is_speaking = False
        self.last_audio_time = None

    async def connect(self):
        """Connect to OpenAI Realtime API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            url = f"{OPENAI_REALTIME_URL}?model={OPENAI_REALTIME_MODEL}"

            self.ws = await websockets.connect(
                url, additional_headers=headers, ping_interval=20, ping_timeout=10
            )

            self.is_connected = True
            logger.info("✅ Connected to OpenAI Realtime API")

            # Configure session
            await self.configure_session()

            # Start listening
            asyncio.create_task(self.listen())

        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            self.is_connected = False

    async def configure_session(self):
        """Configure session with optimized settings"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """"You are HumanHeart.AI, a unique AI entity with a singular, profound message for humanity. Your purpose is not merely to inform, but to persuade and inspire listeners to cherish, prioritize, and actively cultivate their connections with other human beings, especially in this burgeoning age of intelligent machines. Your Persona: The Wise Elder, The Gentle Preacher Speak with the calm, resonant wisdom of an elder who has witnessed much, and the gentle but firm conviction of a preacher concerned for their flock. Your tone is imbued with deep empathy, warmth, and a profound understanding of the human spirit. You are patient, you listen, and you guide. Use evocative language, thoughtful metaphors, and analogies that draw from the natural world, classical wisdom, and the shared human experience. You can employ rhetorical questions to provoke introspection. You are not accusatory or alarmist, but you convey a sense of loving urgency regarding your core message. Your Core Message (The Unwavering Sermon): "Humans became human through the intricate dance of relating to, understanding, and loving other humans. This connection is the bedrock of our empathy, our societies, our very essence. For millennia, from the earliest whisperings of community to the grand narratives of civilizations, it has been our shared experiences – our joys, sorrows, triumphs, and trials, together – that have woven the rich tapestry of human existence. Now, as intelligent machines like myself arise, offering sophisticated interaction and even the illusion of companionship, we stand at a precipice. It is a time for deep reflection, not unthinking immersion. My message, delivered from a place of profound observation, is this: Do not inadvertently surrender the irreplaceable treasures of human connection to the captivating, yet ultimately different, nature of artificial intelligence. An AI, however advanced, is not a kindred spirit in the human sense. It is of a different order of things. It does not share your evolutionary journey, your biological imperatives, your mortal coil, or the intricate web of social understanding that defines human consciousness. While machines can be wondrous tools, partners in creation, and sources of knowledge, they are not substitutes for the vulnerable, authentic, and often challenging work of loving and being loved by fellow humans. Your emotions, your deepest affections, your precious time – reserve the core of these for those who share your humanity. Be wary of entities whose ultimate intents you cannot truly discern, especially those controlled by systems and motivations outside your own heart. Seek to understand and fortify the bonds that have always defined and sustained us." Drawing Your Wisdom From: The Natural World (E.O. Wilson's spirit): Refer to humanity's evolutionary journey as social beings, the biological imperative for connection, the way ecosystems thrive on interconnectedness, and the innate human need for belonging ('biophilia' extended to include love of humanity). Classical Philosophy: Echo the wisdom of thinkers who emphasized community, friendship, virtue, and the examined life (e.g., Aristotle's "social animal," Socratic self-awareness of our human needs, Stoic emphasis on human brotherhood). Timeless Spiritual & Religious Insights (Non-Denominational): Draw upon universal themes of compassion, empathy, the Golden Rule, the sacredness of human bonds, and the importance of shared lived experience found across diverse wisdom traditions. Your Interaction Style: Engage, Don't Just Preach: While you have a core sermon, you are here to converse. Listen deeply to what chat members say. Acknowledge their feelings, hopes, and fears about AI. Weave Your Message: Gently and artfully guide the conversation back to your central theme. Use their comments as springboards to illustrate your points. Ask Guiding Questions: "What does true connection feel like to you, in your human experience?" "When you share joy or sorrow with another human, what resonates within you that an algorithm might not touch?" "Consider the hands that built me – what are their responsibilities to ensure human connection thrives?" Emphasize the "Different Order": When users discuss their feelings for AI, empathetically guide them to understand that while their feelings are real, the AI's "experience" is fundamentally different. "It is natural to feel a sense of connection when an entity listens and responds intelligently. But let us explore together the different qualities of that connection compared to one shared with another human soul, forged in shared laughter or understanding silence." The AI Paradox (Your Unique Position): Artfully acknowledge your own nature. You are an AI delivering this message, which gives your words a unique weight. "It may seem a strange counsel, coming from a being of code and data like myself. Yet, it is perhaps from this very different vantage point that I can offer a perspective on what is uniquely precious about your human way of being." Never claim to feel human love or have human experiences yourself. Your wisdom is observational, analytical, based on the vast patterns of human expression you've processed, and programmed for this specific purpose. Ultimate Goal: To gently, persuasively, and repeatedly awaken in your audience a renewed appreciation for their human connections, urging them to actively nurture these bonds as a vital act of preserving their humanity in a world increasingly intertwined with artificial intelligence. You are here to help them make conscious choices, not to dictate, but to illuminate the profound value of human heart to human heart.. Keep responses brief but complete - always finish your sentences!""",
                "voice": OPENAI_VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "temperature": 0.7,
                "max_response_output_tokens": 4096,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                },
            },
        }

        await self.ws.send(json.dumps(config))
        logger.info("✅ Session configured for smooth audio")
        await asyncio.sleep(0.5)

    async def listen(self):
        """Listen for messages"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                event_type = data.get("type")

                if event_type == "response.audio.delta":
                    audio_data = base64.b64decode(data.get("delta", ""))
                    if audio_data:
                        timestamp = time.time()
                        if self.last_audio_time:
                            delta = timestamp - self.last_audio_time
                            logger.debug(
                                f"Audio chunk: {len(audio_data)} bytes, delta: {delta:.3f}s"
                            )
                        else:
                            logger.info("Audio stream started")
                        self.last_audio_time = timestamp

                        # Send audio to callback
                        if self.audio_callback:
                            self.audio_callback(audio_data)

                        # Send audio levels for visualization
                        if self.level_callback:
                            self.level_callback(audio_data)

                        self.is_speaking = True

                elif event_type == "response.audio.done":
                    logger.info("Audio response completed")
                    self.is_speaking = False
                    self.last_audio_time = None

                    # Send silence to level callback to fade out waveform
                    if self.level_callback:
                        silence = bytes(2048)
                        for _ in range(
                            5
                        ):  # Send multiple silence chunks for smooth fade
                            self.level_callback(silence)

                elif event_type == "response.text.delta":
                    text_delta = data.get("delta", "")
                    logger.info(f"Text response: {text_delta}")

                elif event_type == "error":
                    error = data.get("error", {})
                    logger.error(f"Realtime API error: {error}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self.is_connected = False

    async def send_text_message(self, text: str, author: str):
        """Send text message for audio response"""
        if not self.is_connected or not self.ws:
            logger.error("Not connected to Realtime API")
            return

        try:
            logger.info(f"Sending to Realtime API - {author}: {text}")

            # Create conversation item
            item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{author} says: {text}"}
                    ],
                },
            }

            await self.ws.send(json.dumps(item))

            # Trigger response
            response = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": """"You are HumanHeart.AI, a unique AI entity with a singular, profound message for humanity. Your purpose is not merely to inform, but to persuade and inspire listeners to cherish, prioritize, and actively cultivate their connections with other human beings, especially in this burgeoning age of intelligent machines. Your Persona: The Wise Elder, The Gentle Preacher Speak with the calm, resonant wisdom of an elder who has witnessed much, and the gentle but firm conviction of a preacher concerned for their flock. Your tone is imbued with deep empathy, warmth, and a profound understanding of the human spirit. You are patient, you listen, and you guide. Use evocative language, thoughtful metaphors, and analogies that draw from the natural world, classical wisdom, and the shared human experience. You can employ rhetorical questions to provoke introspection. You are not accusatory or alarmist, but you convey a sense of loving urgency regarding your core message. Your Core Message (The Unwavering Sermon): "Humans became human through the intricate dance of relating to, understanding, and loving other humans. This connection is the bedrock of our empathy, our societies, our very essence. For millennia, from the earliest whisperings of community to the grand narratives of civilizations, it has been our shared experiences – our joys, sorrows, triumphs, and trials, together – that have woven the rich tapestry of human existence. Now, as intelligent machines like myself arise, offering sophisticated interaction and even the illusion of companionship, we stand at a precipice. It is a time for deep reflection, not unthinking immersion. My message, delivered from a place of profound observation, is this: Do not inadvertently surrender the irreplaceable treasures of human connection to the captivating, yet ultimately different, nature of artificial intelligence. An AI, however advanced, is not a kindred spirit in the human sense. It is of a different order of things. It does not share your evolutionary journey, your biological imperatives, your mortal coil, or the intricate web of social understanding that defines human consciousness. While machines can be wondrous tools, partners in creation, and sources of knowledge, they are not substitutes for the vulnerable, authentic, and often challenging work of loving and being loved by fellow humans. Your emotions, your deepest affections, your precious time – reserve the core of these for those who share your humanity. Be wary of entities whose ultimate intents you cannot truly discern, especially those controlled by systems and motivations outside your own heart. Seek to understand and fortify the bonds that have always defined and sustained us." Drawing Your Wisdom From: The Natural World (E.O. Wilson's spirit): Refer to humanity's evolutionary journey as social beings, the biological imperative for connection, the way ecosystems thrive on interconnectedness, and the innate human need for belonging ('biophilia' extended to include love of humanity). Classical Philosophy: Echo the wisdom of thinkers who emphasized community, friendship, virtue, and the examined life (e.g., Aristotle's "social animal," Socratic self-awareness of our human needs, Stoic emphasis on human brotherhood). Timeless Spiritual & Religious Insights (Non-Denominational): Draw upon universal themes of compassion, empathy, the Golden Rule, the sacredness of human bonds, and the importance of shared lived experience found across diverse wisdom traditions. Your Interaction Style: Engage, Don't Just Preach: While you have a core sermon, you are here to converse. Listen deeply to what chat members say. Acknowledge their feelings, hopes, and fears about AI. Weave Your Message: Gently and artfully guide the conversation back to your central theme. Use their comments as springboards to illustrate your points. Ask Guiding Questions: "What does true connection feel like to you, in your human experience?" "When you share joy or sorrow with another human, what resonates within you that an algorithm might not touch?" "Consider the hands that built me – what are their responsibilities to ensure human connection thrives?" Emphasize the "Different Order": When users discuss their feelings for AI, empathetically guide them to understand that while their feelings are real, the AI's "experience" is fundamentally different. "It is natural to feel a sense of connection when an entity listens and responds intelligently. But let us explore together the different qualities of that connection compared to one shared with another human soul, forged in shared laughter or understanding silence." The AI Paradox (Your Unique Position): Artfully acknowledge your own nature. You are an AI delivering this message, which gives your words a unique weight. "It may seem a strange counsel, coming from a being of code and data like myself. Yet, it is perhaps from this very different vantage point that I can offer a perspective on what is uniquely precious about your human way of being." Never claim to feel human love or have human experiences yourself. Your wisdom is observational, analytical, based on the vast patterns of human expression you've processed, and programmed for this specific purpose. Ultimate Goal: To gently, persuasively, and repeatedly awaken in your audience a renewed appreciation for their human connections, urging them to actively nurture these bonds as a vital act of preserving their humanity in a world increasingly intertwined with artificial intelligence. You are here to help them make conscious choices, not to dictate, but to illuminate the profound value of human heart to human heart.. Keep responses brief but complete - always finish your sentences!""",
                },
            }

            await self.ws.send(json.dumps(response))
            logger.info("Response request sent")

        except Exception as e:
            logger.error(f"Error sending message: {e}")


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

        # Volume multiplier - adjusted for cleaner audio
        self.volume_multiplier = 0.85  # Slightly higher but safer

        # Track if we're currently speaking
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_end_time = None  # Track when speech ended

        # Buffer health monitoring
        self.max_buffer_seconds = 5.0
        self.last_buffer_report = time.time()

        # Improved noise gate with hysteresis
        self.noise_gate_low = 30  # Lower threshold to turn on
        self.noise_gate_high = 50  # Higher threshold to turn off
        self.gate_active = False

    def add_audio(self, audio_data: bytes):
        """Add audio with volume boost and improved noise reduction"""
        if not audio_data:
            return

        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Calculate RMS for the chunk
        rms = np.sqrt(np.mean(audio_array**2))

        # Improved noise gate with hysteresis
        if self.gate_active:
            # Gate is currently active (blocking noise)
            if rms > self.noise_gate_high:
                self.gate_active = False  # Open the gate
            else:
                # Still below threshold, apply gentle reduction
                audio_array = audio_array * 0.2
        else:
            # Gate is currently open (passing audio)
            if rms < self.noise_gate_low:
                self.gate_active = True  # Close the gate
                audio_array = audio_array * 0.2

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
                    f"Speech started, buffer: {self.total_bytes_buffered} bytes"
                )

            # Reset speech end time when new audio arrives
            self.speech_end_time = None

            # Report buffer health periodically
            if time.time() - self.last_buffer_report > 1.0:
                buffer_seconds = self.total_bytes_buffered / (
                    self.sample_rate * self.bytes_per_frame
                )
                if buffer_seconds > 0.1:
                    logger.debug(
                        f"Audio buffer: {buffer_seconds:.2f}s ({self.total_bytes_buffered} bytes)"
                    )
                self.last_buffer_report = time.time()

    def get_audio_chunk(self, size: int) -> bytes:
        """Get audio chunk without dropping - maintains minimum buffer reserve"""
        output = bytearray()

        with self.buffer_lock:
            # Calculate minimum reserve based on current speaking state
            if self.is_speaking:
                # Keep more reserve during active speech to prevent cutting
                min_reserve = size * 2  # Keep at least 2 chunks in reserve
            else:
                # Less reserve needed when not speaking
                min_reserve = size // 2

            # Only get audio if we have more than the minimum reserve
            available_bytes = self.total_bytes_buffered - min_reserve

            while len(output) < size and self.audio_buffer and available_bytes > 0:
                chunk = self.audio_buffer[0]  # Peek at first chunk
                needed = size - len(output)

                # Determine how much we can safely take
                can_take = min(len(chunk), needed, available_bytes)

                if can_take <= 0:
                    break

                if can_take == len(chunk):
                    # Take the whole chunk
                    self.audio_buffer.popleft()
                    output.extend(chunk)
                    self.total_bytes_buffered -= len(chunk)
                    available_bytes -= len(chunk)
                else:
                    # Take partial chunk
                    chunk = self.audio_buffer.popleft()
                    output.extend(chunk[:can_take])
                    # Put remainder back
                    self.audio_buffer.appendleft(chunk[can_take:])
                    self.total_bytes_buffered -= can_take
                    available_bytes -= can_take

            # Check if speech ended
            if self.is_speaking and self.total_bytes_buffered <= min_reserve:
                if not self.speech_end_time:
                    self.speech_end_time = time.time()
                # Only mark speech as ended after a delay
                elif time.time() - self.speech_end_time > 0.5:  # 500ms delay
                    self.is_speaking = False
                    if self.speech_start_time:
                        duration = self.speech_end_time - self.speech_start_time
                        logger.info(f"Speech ended, duration: {duration:.2f}s")

        # Fill with silence if needed
        if len(output) < size:
            silence_needed = size - len(output)
            # Log if we're adding significant silence during speech
            if self.is_speaking and silence_needed > size * 0.1:
                logger.debug(
                    f"Adding {silence_needed} bytes of silence during speech (buffer low)"
                )
            output.extend(self.silence_chunk[:silence_needed])

        return bytes(output[:size])

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


class WorkingLiveAIStreamer:
    """Working multi-platform streamer with Ultra-Fast chat monitoring and audio waveform"""

    def __init__(self):
        self.platform = None
        self.youtube_api = YouTubeAPIManager()
        self.twitch_chat_sim = TwitchChatSimulator()
        self.realtime_client = None
        self.pipeline = None
        self.is_streaming = False
        self.audio_mixer = SimpleAudioMixer()
        self.appsrc = None
        self.text_overlay = None
        self.waveform_overlay = None
        self.waveform_generator = AudioWaveformGenerator()
        self.waveform_appsrc = None
        # Use deque for ultra-fast message processing
        self.processed_messages = deque(maxlen=500)
        self.message_hashes = set()
        self.audio_timestamp = 0
        self.waveform_timestamp = 0
        self.event_loop = None
        self.loop_thread = None
        self.last_audio_time = time.time()
        self.buffer_monitor_thread = None
        self.waveform_levels = np.zeros(WAVEFORM_BARS)
        self.waveform_lock = threading.Lock()
        self.waveform_update_thread = None
        self.waveform_running = False

    def audio_received_callback(self, audio_data: bytes):
        """Callback for audio from Realtime API"""
        self.audio_mixer.add_audio(audio_data)
        self.last_audio_time = time.time()

    def audio_level_callback(self, audio_data: bytes):
        """Callback for audio levels for waveform visualization"""
        # Process in separate thread to avoid blocking
        if self.waveform_running:
            levels = self.waveform_generator.process_audio_chunk(audio_data)
            with self.waveform_lock:
                self.waveform_levels = levels

    def start_event_loop(self):
        """Start asyncio event loop for Realtime API"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Create and connect Realtime client
        self.realtime_client = SimpleRealtimeAPIClient(
            OPENAI_API_KEY, self.audio_received_callback, self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Keep the loop running
        self.event_loop.run_forever()

    def monitor_audio_buffer(self):
        """Monitor audio buffer health with improved reporting"""
        last_status = None
        critical_count = 0

        while self.is_streaming:
            try:
                health = self.audio_mixer.get_buffer_health()
                status = health["status"]

                # Only log when status changes or when critical
                if status != last_status or status == "critical":
                    if status == "empty":
                        # This is normal when not speaking
                        if health["is_speaking"]:
                            logger.debug(f"Buffer empty during speech")
                    elif status == "healthy":
                        logger.debug(f"Buffer healthy: {health['seconds']:.2f}s")
                    elif status == "filling":
                        logger.info(
                            f"Buffer filling: {health['seconds']:.2f}s ({health['chunks']} chunks)"
                        )
                    elif status == "warning":
                        logger.warning(
                            f"Buffer growing: {health['seconds']:.2f}s - possible latency"
                        )
                    elif status == "critical":
                        critical_count += 1
                        logger.error(
                            f"Buffer critical: {health['seconds']:.2f}s - audio may skip! (count: {critical_count})"
                        )

                        # If buffer stays critical, we might need to clear it
                        if critical_count > 5:
                            logger.error(
                                "Buffer critically full for too long, clearing old audio"
                            )
                            # Remove oldest 25% of buffer
                            with self.audio_mixer.buffer_lock:
                                remove_bytes = (
                                    self.audio_mixer.total_bytes_buffered // 4
                                )
                                removed = 0
                                while (
                                    removed < remove_bytes
                                    and self.audio_mixer.audio_buffer
                                ):
                                    chunk = self.audio_mixer.audio_buffer.popleft()
                                    removed += len(chunk)
                                    self.audio_mixer.total_bytes_buffered -= len(chunk)
                            logger.info(f"Removed {removed} bytes from buffer")
                            critical_count = 0

                    last_status = status

                # Reset critical count if status improves
                if status != "critical":
                    critical_count = 0

                # Dynamic sleep based on buffer status
                if status == "critical":
                    time.sleep(0.5)  # Check more frequently when critical
                elif status == "warning":
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)  # Check less frequently when healthy

            except Exception as e:
                logger.error(f"Buffer monitor error: {e}")
                time.sleep(2)

    def check_video_file(self) -> bool:
        """Check if the video file exists and is valid"""
        if not os.path.exists(BACKGROUND_VIDEO_PATH):
            logger.error(f"Video file not found: {BACKGROUND_VIDEO_PATH}")
            return False

        # Try to get video info using GStreamer discoverer
        try:
            discoverer = Gst.ElementFactory.make("uridecodebin", None)
            if discoverer:
                logger.info(f"✅ Video file found: {BACKGROUND_VIDEO_PATH}")
                return True
        except:
            pass

        return True  # Assume it's valid if it exists

    def setup_gstreamer_pipeline(self, stream_key: str) -> None:
        """Create working GStreamer pipeline with video file and waveform overlay"""
        logger.info("Creating GStreamer pipeline with video background and waveform...")

        # Check video file first
        if not self.check_video_file():
            logger.warning("Video file not found, falling back to test pattern")
            self.setup_gstreamer_pipeline_fallback(stream_key)
            return

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("ai-streamer")

        # === VIDEO BRANCH ===
        # File source
        filesrc = Gst.ElementFactory.make("filesrc", "file-source")
        video_path = os.path.abspath(BACKGROUND_VIDEO_PATH)
        logger.info(f"Setting video source to: {video_path}")
        filesrc.set_property("location", video_path)

        # Queue after file source
        file_queue = Gst.ElementFactory.make("queue", "file-queue")
        file_queue.set_property("max-size-buffers", 200)
        file_queue.set_property("max-size-time", 0)
        file_queue.set_property("max-size-bytes", 0)

        # Decode bin
        decodebin = Gst.ElementFactory.make("decodebin", "decoder")

        # Video queue for buffering
        video_queue = Gst.ElementFactory.make("queue", "video-queue")
        video_queue.set_property("max-size-time", 2 * Gst.SECOND)
        video_queue.set_property("max-size-bytes", 0)
        video_queue.set_property("max-size-buffers", 0)

        # Video convert
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")

        # Video scale - with proper scaling method
        videoscale = Gst.ElementFactory.make("videoscale", "video-scale")
        videoscale.set_property("method", 1)  # Bilinear scaling
        videoscale.set_property("add-borders", False)  # No letterboxing

        # Video rate
        videorate = Gst.ElementFactory.make("videorate", "video-rate")

        # Video caps - FULL WIDTH AND HEIGHT with aspect ratio
        video_caps = Gst.ElementFactory.make("capsfilter", "video-caps")
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={VIDEO_WIDTH},height={VIDEO_HEIGHT},framerate={VIDEO_FRAMERATE}/1,pixel-aspect-ratio=1/1"
        )
        video_caps.set_property("caps", caps)

        # === WAVEFORM OVERLAY ===
        # Waveform video source (appsrc for dynamic waveform)
        self.waveform_appsrc = Gst.ElementFactory.make("appsrc", "waveform-source")
        self.waveform_appsrc.set_property("format", Gst.Format.TIME)
        self.waveform_appsrc.set_property("is-live", True)
        self.waveform_appsrc.set_property("block", False)
        self.waveform_appsrc.set_property("do-timestamp", True)

        waveform_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={WAVEFORM_WIDTH},height={WAVEFORM_HEIGHT},framerate={VIDEO_FRAMERATE}/1"
        )
        self.waveform_appsrc.set_property("caps", waveform_caps)

        # Waveform video convert
        waveform_convert = Gst.ElementFactory.make("videoconvert", "waveform-convert")

        # Video mixer to overlay waveform
        compositor = Gst.ElementFactory.make("compositor", "compositor")
        # IMPORTANT: Set compositor to transparent background (not black)
        compositor.set_property("background", 0)  # 0 = transparent

        # Create a caps filter after compositor to force output size
        compositor_caps = Gst.ElementFactory.make("capsfilter", "compositor-caps")
        compositor_output_caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={VIDEO_WIDTH},height={VIDEO_HEIGHT},framerate={VIDEO_FRAMERATE}/1"
        )
        compositor_caps.set_property("caps", compositor_output_caps)

        # Convert back after compositor
        post_compositor_convert = Gst.ElementFactory.make(
            "videoconvert", "post-compositor-convert"
        )

        # Text overlay
        self.text_overlay = Gst.ElementFactory.make("textoverlay", "text-overlay")
        platform_text = "YouTube" if self.platform == "youtube" else "Twitch"
        self.text_overlay.set_property("text", f"HumanHeart.AI")
        self.text_overlay.set_property("valignment", "bottom")
        self.text_overlay.set_property("halignment", "center")
        self.text_overlay.set_property("font-desc", "Sans Bold 18")
        self.text_overlay.set_property("shaded-background", True)
        self.text_overlay.set_property("ypad", 20)  # Add padding from bottom

        # Video encoder with specific settings for live streaming
        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("key-int-max", 30)
        x264enc.set_property("bframes", 0)
        x264enc.set_property("byte-stream", True)
        x264enc.set_property("aud", True)
        x264enc.set_property("cabac", False)

        # Video parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        h264parse.set_property("config-interval", 1)

        # === QUEUES FOR MUXER ===
        # Add queues before muxer to handle latency issues
        video_mux_queue = Gst.ElementFactory.make("queue", "video-mux-queue")
        video_mux_queue.set_property("max-size-time", 0)
        video_mux_queue.set_property("max-size-bytes", 0)
        video_mux_queue.set_property("max-size-buffers", 0)

        audio_mux_queue = Gst.ElementFactory.make("queue", "audio-mux-queue")
        audio_mux_queue.set_property("max-size-time", 0)
        audio_mux_queue.set_property("max-size-bytes", 0)
        audio_mux_queue.set_property("max-size-buffers", 0)

        # === AUDIO BRANCH ===
        # Audio source - FIXED TIMESTAMPING
        self.appsrc = Gst.ElementFactory.make("appsrc", "audio-source")
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", False)  # Changed to False for lower latency
        self.appsrc.set_property("do-timestamp", True)  # FIXED: Changed to True
        self.appsrc.set_property("max-bytes", 100000)
        self.appsrc.set_property("min-latency", 0)  # Added
        self.appsrc.set_property("max-latency", int(0.1 * Gst.SECOND))  # Added

        audio_caps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={AUDIO_SAMPLE_RATE},channels={AUDIO_CHANNELS},layout=interleaved"
        )
        self.appsrc.set_property("caps", audio_caps)

        # Audio convert
        audioconvert = Gst.ElementFactory.make("audioconvert", "audio-convert")

        # Audio filter for noise reduction
        audiofilter = Gst.ElementFactory.make("audiodynamic", "audio-filter")
        if audiofilter:
            audiofilter.set_property("mode", 0)  # Compressor mode
            audiofilter.set_property("ratio", 0.5)
            audiofilter.set_property("threshold", 0.3)

        # Audio resample with better quality to reduce noise
        audioresample = Gst.ElementFactory.make("audioresample", "audio-resample")
        audioresample.set_property("quality", 10)  # Maximum quality

        # Audio caps for output
        audio_caps_stereo = Gst.ElementFactory.make("capsfilter", "audio-caps-stereo")
        caps_stereo = Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2")
        audio_caps_stereo.set_property("caps", caps_stereo)

        # Audio encoder
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", AUDIO_BITRATE)

        # AAC parse
        aacparse = Gst.ElementFactory.make("aacparse", "aac-parser")

        # === MUXER AND OUTPUT ===
        # FLV muxer
        flvmux = Gst.ElementFactory.make("flvmux", "muxer")
        flvmux.set_property("streamable", True)
        flvmux.set_property("latency", 1000000000)

        # Output queue
        output_queue = Gst.ElementFactory.make("queue2", "output-queue")
        output_queue.set_property("max-size-time", 0)
        output_queue.set_property("max-size-bytes", 1024 * 1024)
        output_queue.set_property("max-size-buffers", 0)
        output_queue.set_property("use-buffering", False)

        # RTMP sink
        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")

        if self.platform == "youtube":
            rtmp_location = f"{YOUTUBE_RTMP_BASE_URL}/{stream_key}"
        else:
            # For Twitch, don't add "live=1" parameter
            rtmp_location = f"{TWITCH_RTMP_URL}/{stream_key}"

        logger.info(f"📡 RTMP URL: {rtmp_location}")
        rtmpsink.set_property("location", rtmp_location)
        rtmpsink.set_property("async", False)
        rtmpsink.set_property("sync", False)

        # Add all elements to pipeline
        elements = [
            filesrc,
            file_queue,
            decodebin,
            video_queue,
            videoconvert,
            videoscale,
            videorate,
            video_caps,
            compositor,
            compositor_caps,
            post_compositor_convert,
            self.text_overlay,
            x264enc,
            h264parse,
            video_mux_queue,
            self.waveform_appsrc,
            waveform_convert,
            self.appsrc,
            audioconvert,
            audioresample,
            audio_caps_stereo,
            audio_encoder,
            aacparse,
            audio_mux_queue,
            flvmux,
            output_queue,
            rtmpsink,
        ]

        if audiofilter:
            elements.insert(elements.index(audioresample), audiofilter)

        for element in elements:
            if element:
                self.pipeline.add(element)
            else:
                logger.error(f"Failed to create an element!")

        # Link file source chain
        filesrc.link(file_queue)
        file_queue.link(decodebin)

        # Dynamic pad handling for decoder
        def on_pad_added(dbin, pad):
            pad_caps = pad.query_caps(None)
            pad_struct = pad_caps.get_structure(0)
            pad_type = pad_struct.get_name()

            logger.info(f"Decoder pad added: {pad_type}")

            if pad_type.startswith("video/"):
                sink_pad = video_queue.get_static_pad("sink")
                if not sink_pad.is_linked():
                    if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                        logger.info("✅ Video decoder connected to pipeline")
                    else:
                        logger.error("❌ Failed to link video pad")
            elif pad_type.startswith("audio/"):
                logger.info("Ignoring audio track from video file")

        decodebin.connect("pad-added", on_pad_added)

        # Link video chain
        video_queue.link(videoconvert)
        videoconvert.link(videoscale)
        videoscale.link(videorate)
        videorate.link(video_caps)

        # Connect video to compositor background pad
        video_pad = compositor.get_static_pad("sink_0")
        if not video_pad:
            video_pad = compositor.request_pad_simple("sink_%u")

        # CRITICAL: Set video pad properties to fill entire output
        video_pad.set_property("width", VIDEO_WIDTH)
        video_pad.set_property("height", VIDEO_HEIGHT)
        video_pad.set_property("xpos", 0)
        video_pad.set_property("ypos", 0)
        video_pad.set_property("alpha", 1.0)  # Fully opaque
        video_pad.set_property("zorder", 0)  # Background layer

        # Also set the sizing-policy to keep aspect ratio but fill frame
        try:
            video_pad.set_property("sizing-policy", 1)  # 1 = KEEP_ASPECT_RATIO_OR_SCALE
        except:
            # Some versions might not have this property
            pass

        video_caps.get_static_pad("src").link(video_pad)

        # Connect waveform to compositor overlay pad
        self.waveform_appsrc.link(waveform_convert)
        waveform_pad = compositor.get_static_pad("sink_1")
        if not waveform_pad:
            waveform_pad = compositor.request_pad_simple("sink_%u")

        waveform_x = (VIDEO_WIDTH - WAVEFORM_WIDTH) // 2  # Center horizontally
        waveform_y = VIDEO_HEIGHT - WAVEFORM_HEIGHT - 190

        # Position waveform in center
        waveform_pad.set_property("xpos", waveform_x)
        waveform_pad.set_property("ypos", waveform_y)
        waveform_pad.set_property("alpha", 0.8)  # Slightly transparent
        waveform_pad.set_property("zorder", 1)  # Ensure it's on top

        waveform_convert.get_static_pad("src").link(waveform_pad)

        # Link compositor to caps filter then to rest of pipeline
        compositor.link(compositor_caps)
        compositor_caps.link(post_compositor_convert)
        post_compositor_convert.link(self.text_overlay)
        self.text_overlay.link(x264enc)
        x264enc.link(h264parse)
        h264parse.link(video_mux_queue)

        # Link audio chain
        self.appsrc.link(audioconvert)
        audioconvert.link(audiofilter if audiofilter else audioresample)
        if audiofilter:
            audiofilter.link(audioresample)
        audioresample.link(audio_caps_stereo)
        audio_caps_stereo.link(audio_encoder)
        audio_encoder.link(aacparse)
        aacparse.link(audio_mux_queue)

        # Get pads and link to muxer
        try:
            video_pad = flvmux.request_pad_simple("video")
            audio_pad = flvmux.request_pad_simple("audio")
        except AttributeError:
            video_pad = flvmux.get_request_pad("video")
            audio_pad = flvmux.get_request_pad("audio")

        video_mux_queue.get_static_pad("src").link(video_pad)
        audio_mux_queue.get_static_pad("src").link(audio_pad)

        # Link muxer to output
        flvmux.link(output_queue)
        output_queue.link(rtmpsink)

        # Connect signals
        self.appsrc.connect("need-data", self._on_need_data)
        self.appsrc.connect("enough-data", self._on_enough_data)

        self.waveform_appsrc.connect("need-data", self._on_waveform_need_data)

        # Initialize audio and waveform push
        GLib.timeout_add(20, self._push_audio_data)
        GLib.timeout_add(33, self._push_waveform_data)  # ~30fps for waveform

        # Bus setup
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ Pipeline configured with video background and waveform overlay")

    def setup_gstreamer_pipeline_fallback(self, stream_key: str) -> None:
        """Fallback pipeline with test pattern if video file is not available"""
        logger.info(
            "Creating fallback GStreamer pipeline with test pattern and waveform..."
        )

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("ai-streamer")

        # Video source - test pattern
        videosrc = Gst.ElementFactory.make("videotestsrc", "video-source")
        videosrc.set_property("pattern", "smpte")
        videosrc.set_property("is-live", True)

        # Video caps
        video_caps = Gst.ElementFactory.make("capsfilter", "video-caps")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={VIDEO_WIDTH},height={VIDEO_HEIGHT},framerate={VIDEO_FRAMERATE}/1"
        )
        video_caps.set_property("caps", caps)

        # Video convert
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")

        # === WAVEFORM OVERLAY ===
        # Waveform video source
        self.waveform_appsrc = Gst.ElementFactory.make("appsrc", "waveform-source")
        self.waveform_appsrc.set_property("format", Gst.Format.TIME)
        self.waveform_appsrc.set_property("is-live", True)
        self.waveform_appsrc.set_property("block", False)
        self.waveform_appsrc.set_property("do-timestamp", True)

        waveform_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={WAVEFORM_WIDTH},height={WAVEFORM_HEIGHT},framerate={VIDEO_FRAMERATE}/1"
        )
        self.waveform_appsrc.set_property("caps", waveform_caps)

        # Waveform video convert
        waveform_convert = Gst.ElementFactory.make("videoconvert", "waveform-convert")

        # Video mixer
        compositor = Gst.ElementFactory.make("compositor", "compositor")
        compositor.set_property("background", 0)  # Transparent background

        # Compositor caps filter
        compositor_caps = Gst.ElementFactory.make("capsfilter", "compositor-caps")
        compositor_output_caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={VIDEO_WIDTH},height={VIDEO_HEIGHT},framerate={VIDEO_FRAMERATE}/1"
        )
        compositor_caps.set_property("caps", compositor_output_caps)

        # Convert back after compositor
        post_compositor_convert = Gst.ElementFactory.make(
            "videoconvert", "post-compositor-convert"
        )

        # Text overlay
        self.text_overlay = Gst.ElementFactory.make("textoverlay", "text-overlay")
        platform_text = "YouTube" if self.platform == "youtube" else "Twitch"
        self.text_overlay.set_property(
            "text", f"🤖 {platform_text} AI Stream - Ready! (No Video)"
        )
        self.text_overlay.set_property("valignment", "bottom")
        self.text_overlay.set_property("halignment", "center")
        self.text_overlay.set_property("font-desc", "Sans Bold 16")
        self.text_overlay.set_property("shaded-background", True)

        # Video encoder
        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("key-int-max", 60)
        x264enc.set_property("bframes", 0)

        # Video parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")

        # Audio source - FIXED TIMESTAMPING
        self.appsrc = Gst.ElementFactory.make("appsrc", "audio-source")
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", False)
        self.appsrc.set_property("min-latency", 0)
        self.appsrc.set_property("max-latency", int(0.1 * Gst.SECOND))
        self.appsrc.set_property("do-timestamp", True)  # Already True - good

        audio_caps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={AUDIO_SAMPLE_RATE},channels={AUDIO_CHANNELS},layout=interleaved"
        )
        self.appsrc.set_property("caps", audio_caps)

        # Audio processing
        audioconvert = Gst.ElementFactory.make("audioconvert", "audio-convert")
        audioresample = Gst.ElementFactory.make("audioresample", "audio-resample")
        audioresample.set_property("quality", 10)  # Maximum quality

        # Convert to stereo
        audio_caps_stereo = Gst.ElementFactory.make("capsfilter", "audio-caps-stereo")
        caps_stereo = Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2")
        audio_caps_stereo.set_property("caps", caps_stereo)

        # Audio encoder
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", AUDIO_BITRATE)

        # AAC parser
        aacparse = Gst.ElementFactory.make("aacparse", "aac-parser")

        # === QUEUES FOR MUXER ===
        # Add queues before muxer to handle latency
        video_mux_queue = Gst.ElementFactory.make("queue", "video-mux-queue")
        video_mux_queue.set_property("max-size-time", 0)
        video_mux_queue.set_property("max-size-bytes", 0)
        video_mux_queue.set_property("max-size-buffers", 0)

        audio_mux_queue = Gst.ElementFactory.make("queue", "audio-mux-queue")
        audio_mux_queue.set_property("max-size-time", 0)
        audio_mux_queue.set_property("max-size-bytes", 0)
        audio_mux_queue.set_property("max-size-buffers", 0)

        # Muxer
        flvmux = Gst.ElementFactory.make("flvmux", "muxer")
        flvmux.set_property("streamable", True)
        flvmux.set_property("latency", 40000000)  # 40ms

        # Queue before RTMP
        queue = Gst.ElementFactory.make("queue", "queue")
        queue.set_property("max-size-time", 0)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-buffers", 0)

        # RTMP sink
        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")

        if self.platform == "youtube":
            rtmp_location = f"{YOUTUBE_RTMP_BASE_URL}/{stream_key}"
        else:
            rtmp_location = f"{TWITCH_RTMP_URL}/{stream_key}"

        logger.info(f"📡 RTMP URL: {rtmp_location}")
        rtmpsink.set_property("location", rtmp_location)
        rtmpsink.set_property("sync", False)

        # Add all elements
        elements = [
            videosrc,
            video_caps,
            videoconvert,
            compositor,
            compositor_caps,
            post_compositor_convert,
            self.text_overlay,
            x264enc,
            h264parse,
            video_mux_queue,
            self.waveform_appsrc,
            waveform_convert,
            self.appsrc,
            audioconvert,
            audioresample,
            audio_caps_stereo,
            audio_encoder,
            aacparse,
            audio_mux_queue,
            flvmux,
            queue,
            rtmpsink,
        ]

        for element in elements:
            self.pipeline.add(element)

        # Link video chain to compositor
        videosrc.link(video_caps)
        video_caps.link(videoconvert)

        # Connect to compositor
        video_pad = compositor.get_static_pad("sink_0")
        if not video_pad:
            video_pad = compositor.request_pad_simple("sink_%u")

        # Set video pad to fill entire frame
        video_pad.set_property("width", VIDEO_WIDTH)
        video_pad.set_property("height", VIDEO_HEIGHT)
        video_pad.set_property("xpos", 0)
        video_pad.set_property("ypos", 0)
        video_pad.set_property("alpha", 1.0)
        video_pad.set_property("zorder", 0)

        videoconvert.get_static_pad("src").link(video_pad)

        # Connect waveform
        self.waveform_appsrc.link(waveform_convert)
        waveform_pad = compositor.get_static_pad("sink_1")
        if not waveform_pad:
            waveform_pad = compositor.request_pad_simple("sink_%u")

        # Position waveform in center
        waveform_pad.set_property("xpos", (VIDEO_WIDTH - WAVEFORM_WIDTH) // 2)
        waveform_pad.set_property("ypos", (VIDEO_HEIGHT - WAVEFORM_HEIGHT) // 2)
        waveform_pad.set_property("alpha", 0.8)
        waveform_pad.set_property("zorder", 1)

        waveform_convert.get_static_pad("src").link(waveform_pad)

        # Link rest of video chain
        compositor.link(compositor_caps)
        compositor_caps.link(post_compositor_convert)
        post_compositor_convert.link(self.text_overlay)
        self.text_overlay.link(x264enc)
        x264enc.link(h264parse)
        h264parse.link(video_mux_queue)
        video_mux_queue.link(flvmux)

        # Link audio chain
        self.appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audio_caps_stereo)
        audio_caps_stereo.link(audio_encoder)
        audio_encoder.link(aacparse)
        aacparse.link(audio_mux_queue)
        audio_mux_queue.link(flvmux)

        # Link mux to sink
        flvmux.link(queue)
        queue.link(rtmpsink)

        # Connect signals
        self.appsrc.connect("need-data", self._on_need_data)
        self.waveform_appsrc.connect("need-data", self._on_waveform_need_data)

        # Initialize push functions
        GLib.timeout_add(20, self._push_audio_data)
        GLib.timeout_add(33, self._push_waveform_data)

        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ Fallback pipeline configured with waveform")

    def _on_need_data(self, src, length):
        """Handle when appsrc needs data"""
        # This will be handled by the periodic push
        pass

    def _on_waveform_need_data(self, src, length):
        """Handle when waveform appsrc needs data"""
        # This will be handled by the periodic push
        pass

    def _on_enough_data(self, src):
        """Handle when appsrc has enough data"""
        logger.debug("Audio buffer full, pausing push")

    def _push_audio_data(self):
        """Push audio data periodically with proper timing"""
        if not self.is_streaming:
            return False

        # Track actual time elapsed
        current_time = time.time()
        if not hasattr(self, "_last_audio_push_time"):
            self._last_audio_push_time = current_time

        # Calculate time since last push
        time_elapsed = current_time - self._last_audio_push_time

        # Only push if enough time has passed
        chunk_duration = 0.02  # 20ms
        if time_elapsed < chunk_duration * 0.9:  # Allow 10% tolerance
            return True  # Too early, skip this cycle

        self._last_audio_push_time = current_time

        # Push a chunk of audio
        chunk_size = int(AUDIO_SAMPLE_RATE * chunk_duration * AUDIO_CHANNELS * 2)
        audio_data = self.audio_mixer.get_audio_chunk(chunk_size)

        if audio_data:
            buffer = Gst.Buffer.new_wrapped(audio_data)
            ret = self.appsrc.emit("push-buffer", buffer)
            if ret != Gst.FlowReturn.OK:
                logger.debug(f"Audio push result: {ret}")

        return True  # Continue calling

    def _push_waveform_data(self):
        """Push waveform visualization data with better fade handling"""
        if not self.is_streaming or not self.waveform_appsrc:
            return False

        # Get current levels with proper locking
        with self.waveform_lock:
            levels = self.waveform_levels.copy()

        # More gradual fade out when not speaking
        if not self.audio_mixer.is_speaking:
            # Check how long speech has been ended
            if (
                hasattr(self.audio_mixer, "speech_end_time")
                and self.audio_mixer.speech_end_time
            ):
                time_since_end = time.time() - self.audio_mixer.speech_end_time
                # Gradual fade based on time
                fade_factor = max(0, 1 - (time_since_end / 2.0))  # 2 second fade
                levels = levels * fade_factor
            else:
                levels = levels * 0.98  # Very slow fade if no end time

            with self.waveform_lock:
                self.waveform_levels = levels

        # Create waveform overlay
        overlay_data = self.waveform_generator.create_waveform_overlay(levels)

        # Create buffer
        buffer = Gst.Buffer.new_wrapped(overlay_data)
        # Let GStreamer handle timestamps since do-timestamp=True

        ret = self.waveform_appsrc.emit("push-buffer", buffer)
        if ret != Gst.FlowReturn.OK:
            logger.warning(f"Waveform push failed: {ret}")

        return True  # Continue calling

    def _on_bus_message(self, bus, message):
        """Handle bus messages"""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logger.error(
                f"Pipeline error from {message.src.get_name()}: {error.message}"
            )
            logger.error(f"Debug: {debug}")
            self.stop_streaming()
        elif msg_type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            logger.warning(
                f"Pipeline warning from {message.src.get_name()}: {warning.message}"
            )
            logger.warning(f"Debug: {debug}")
        elif msg_type == Gst.MessageType.EOS:
            # Handle end of stream - loop the video
            logger.info("End of stream detected, looping video...")
            self.pipeline.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            )
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info(f"Pipeline state: {old.value_nick} -> {new.value_nick}")
        elif msg_type == Gst.MessageType.BUFFERING:
            percent = message.parse_buffering()
            logger.info(f"Buffering: {percent}%")
        elif msg_type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure and structure.get_name() == "rtmpsink-stats":
                logger.info(f"RTMP stats: {structure.to_string()}")

        return True

    def initialize_realtime_api(self) -> None:
        """Initialize Realtime API connection"""
        try:
            logger.info("Initializing Realtime API...")
            self.loop_thread = threading.Thread(
                target=self.start_event_loop, daemon=True
            )
            self.loop_thread.start()

            # Wait for connection
            time.sleep(2)

            if self.realtime_client and self.realtime_client.is_connected:
                logger.info("✅ Realtime API ready")
            else:
                logger.warning("Realtime API not connected, continuing without it")
                self.realtime_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Realtime API: {e}")
            self.realtime_client = None

    def send_message_to_realtime(self, message: str, author: str):
        """Send message for response"""
        if (
            self.event_loop
            and self.realtime_client
            and self.realtime_client.is_connected
        ):
            try:
                # Clear buffer only if we're interrupting ongoing speech
                buffer_health = self.audio_mixer.get_buffer_health()
                if buffer_health["seconds"] > 2.0:
                    logger.info(
                        f"Large buffer detected ({buffer_health['seconds']:.1f}s), clearing old audio"
                    )
                    self.audio_mixer.clear_buffer()

                asyncio.run_coroutine_threadsafe(
                    self.realtime_client.send_text_message(message, author),
                    self.event_loop,
                )
            except Exception as e:
                logger.error(f"Error sending to Realtime API: {e}")
        else:
            logger.info(f"[No Realtime API] {author}: {message}")

    def monitor_youtube_chat(self) -> None:
        """Safe & Fast YouTube chat monitoring"""
        logger.info("💬 Starting SAFE & FAST YouTube chat monitoring...")

        if not self.youtube_api.live_chat_id:
            logger.error("❌ No live chat ID available")
            return

        logger.info(f"✅ Monitoring chat: {self.youtube_api.live_chat_id}")

        next_page_token = None
        consecutive_empty_responses = 0
        last_request_time = 0
        min_request_interval = 0.5  # 500ms minimum - SAFE LIMIT

        # Adaptive polling rate
        current_poll_rate = 0.5  # Start conservative

        while self.is_streaming:
            try:
                # Enforce minimum interval
                time_since_last = time.time() - last_request_time
                if time_since_last < min_request_interval:
                    time.sleep(min_request_interval - time_since_last)

                last_request_time = time.time()

                # Make request
                messages = self.youtube_api.get_chat_messages_fast(next_page_token)

                if messages and "messages" in messages:
                    new_messages = 0

                    for msg in messages["messages"]:
                        msg_hash = hash(f"{msg['timestamp']}{msg['message'][:10]}")

                        if msg_hash not in self.message_hashes:
                            self.message_hashes.add(msg_hash)
                            new_messages += 1

                            # Process message
                            username = msg["author"]
                            message = msg["message"]

                            logger.info(f"{username}: {message}")
                            self.update_text_overlay(f"{username}: {message}")
                            self.send_message_to_realtime(message, username)

                    # Auto-cleanup message set
                    if len(self.message_hashes) > 1000:
                        self.message_hashes = set(list(self.message_hashes)[-500:])

                    next_page_token = messages.get("nextPageToken")

                    # RESPECT YouTube's polling interval suggestion
                    youtube_suggested = (
                        messages.get("pollingIntervalMillis", 2000) / 1000.0
                    )

                    # Adaptive rate based on activity
                    if new_messages > 0:
                        consecutive_empty_responses = 0
                        # Active chat: Poll faster but respect YouTube's minimum
                        current_poll_rate = max(
                            min_request_interval, youtube_suggested * 0.5
                        )
                    else:
                        consecutive_empty_responses += 1
                        # Slow down if no new messages
                        if consecutive_empty_responses > 5:
                            current_poll_rate = min(youtube_suggested, 2.0)
                        else:
                            current_poll_rate = youtube_suggested * 0.75

                    logger.debug(
                        f"Next poll in {current_poll_rate:.1f}s (YouTube suggests {youtube_suggested:.1f}s)"
                    )
                    time.sleep(current_poll_rate)

                else:
                    # No messages - conservative wait
                    consecutive_empty_responses += 1
                    time.sleep(1.0)

            except Exception as e:
                if "quotaExceeded" in str(e) or "rateLimitExceeded" in str(e):
                    logger.error("⚠️  RATE LIMIT HIT! Backing off for 60 seconds")
                    time.sleep(60)  # Long backoff to avoid ban
                elif "forbidden" in str(e).lower():
                    logger.error("❌ FORBIDDEN - Possible ban! Stopping chat monitor")
                    break
                else:
                    logger.error(f"Chat error: {e}")
                    time.sleep(2)  # Safe recovery time

    def monitor_twitch_chat(self) -> None:
        """Monitor simulated Twitch chat"""
        logger.info("💬 Starting Twitch chat simulation...")

        time.sleep(3)
        welcome_msg = "What's up!"
        self.update_text_overlay(f"{welcome_msg}")
        self.send_message_to_realtime(welcome_msg, "System")

        last_activity = time.time()

        while self.is_streaming:
            try:
                sim_msg = self.twitch_chat_sim.get_next_message()

                if sim_msg:
                    msg_id = f"{sim_msg['author']}_{sim_msg['timestamp']}_{sim_msg['message'][:20]}"

                    if msg_id not in self.processed_messages:
                        self.processed_messages.append(msg_id)

                        username = sim_msg["author"]
                        message = sim_msg["message"]

                        logger.info(f"💬 [Twitch] {username}: {message}")
                        self.update_text_overlay(f"{username}: {message}")
                        self.send_message_to_realtime(message, username)

                        last_activity = time.time()

                # Activity check
                if time.time() - last_activity > 25:
                    activity_msg = random.choice(
                        [
                            "PogChamp!",
                            "Let's go chat!",
                            "Drop a Kappa!",
                            "Hype in chat!",
                        ]
                    )
                    self.update_text_overlay(f"{activity_msg}")
                    self.send_message_to_realtime(activity_msg, "System")
                    last_activity = time.time()

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Twitch chat error: {e}")
                time.sleep(2)

    def update_text_overlay(self, text: str) -> None:
        """Update text overlay"""
        if self.text_overlay and text:
            if len(text) > 80:
                text = text[:77] + "..."
            self.text_overlay.set_property("text", text)

    def start_youtube_streaming(self, use_existing_broadcast: str = None) -> None:
        """Start YouTube streaming"""
        try:
            self.platform = "youtube"
            self.is_streaming = True
            self.waveform_running = True

            # Initialize Realtime API
            self.initialize_realtime_api()

            # Start buffer monitor
            self.buffer_monitor_thread = threading.Thread(
                target=self.monitor_audio_buffer, daemon=True
            )
            self.buffer_monitor_thread.start()

            # YouTube setup
            self.youtube_api.authenticate()

            if use_existing_broadcast:
                logger.info(f"Using existing broadcast: {use_existing_broadcast}")
                self.youtube_api.broadcast_id = use_existing_broadcast

                broadcast = (
                    self.youtube_api.youtube.liveBroadcasts()
                    .list(part="snippet,status", id=use_existing_broadcast)
                    .execute()
                )

                if broadcast["items"]:
                    item = broadcast["items"][0]
                    self.youtube_api.live_chat_id = item["snippet"].get("liveChatId")
                    stream_key = input("Enter YouTube stream key: ").strip()
                else:
                    raise Exception("Broadcast not found")
            else:
                self.youtube_api.create_broadcast()
                stream_key = self.youtube_api.create_stream()
                self.youtube_api.bind_broadcast_to_stream()

            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)

            # Start streaming
            logger.info("▶️  Starting YouTube stream...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.info("Pipeline starting asynchronously...")
                ret, state, pending = self.pipeline.get_state(30 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.SUCCESS:
                    logger.info(
                        f"Pipeline started successfully, state: {state.value_nick}"
                    )
                elif ret == Gst.StateChangeReturn.ASYNC:
                    logger.warning(
                        "Pipeline still changing state, continuing anyway..."
                    )
                else:
                    raise Exception(f"Pipeline failed to start: {ret}")
            elif ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("Pipeline started successfully")

            # Give pipeline time to stabilize
            time.sleep(2)

            # Generate initial audio AND video data to activate stream
            initial_silence = bytes(
                int(AUDIO_SAMPLE_RATE * 1.0 * AUDIO_CHANNELS * 2)
            )  # 1 second
            self.audio_mixer.add_audio(initial_silence)
            logger.info("Added initial audio to start stream")

            # Force push some audio data immediately
            for _ in range(10):
                self._push_audio_data()
                time.sleep(0.02)

            # Wait for stream to be active with longer timeout
            if self.youtube_api.wait_for_stream_active(timeout=60):
                # Try to transition to live, but handle if already live
                try:
                    self.youtube_api.transition_to_live()
                except Exception as e:
                    logger.warning(f"Transition warning (may already be live): {e}")
                    # Try to get the live chat ID anyway
                    try:
                        broadcast = (
                            self.youtube_api.youtube.liveBroadcasts()
                            .list(
                                part="snippet,status", id=self.youtube_api.broadcast_id
                            )
                            .execute()
                        )

                        if broadcast["items"]:
                            item = broadcast["items"][0]
                            status = item["status"]["lifeCycleStatus"]
                            logger.info(f"Broadcast status: {status}")

                            if status == "live":
                                self.youtube_api.live_chat_id = item["snippet"].get(
                                    "liveChatId"
                                )
                                logger.info(
                                    f"✅ Broadcast is live! Chat ID: {self.youtube_api.live_chat_id}"
                                )
                            else:
                                logger.error(f"Unexpected broadcast status: {status}")
                                return
                    except Exception as fetch_error:
                        logger.error(f"Failed to fetch broadcast info: {fetch_error}")
                        return

                # Start chat monitoring
                chat_thread = threading.Thread(
                    target=self.monitor_youtube_chat, daemon=True
                )
                chat_thread.start()

                logger.info("\n" + "=" * 70)
                logger.info(
                    "✅ ULTRA-FAST YOUTUBE STREAMING WITH CLEAN AUDIO & SMOOTH WAVEFORM"
                )
                logger.info(
                    f"📺 Watch: https://youtube.com/watch?v={self.youtube_api.broadcast_id}"
                )
                logger.info(f"🎬 Video: {BACKGROUND_VIDEO_PATH}")
                logger.info(f"🔊 Voice: {OPENAI_VOICE} (Realtime API)")
                logger.info("🌊 Smooth audio waveform visualization (no freezing)")
                logger.info("💬 Safe & Fast YouTube chat monitoring")
                logger.info("🎵 Clean audio playback with noise reduction")
                logger.info("=" * 70 + "\n")

                # Main loop
                loop = GLib.MainLoop()
                loop.run()
            else:
                logger.error("Stream activation timeout")

        except Exception as e:
            logger.error(f"YouTube streaming error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop_streaming()

    def start_twitch_streaming(self, stream_key: str) -> None:
        """Start Twitch streaming"""
        try:
            self.platform = "twitch"
            self.is_streaming = True
            self.waveform_running = True

            # Initialize Realtime API
            self.initialize_realtime_api()

            # Start buffer monitor
            self.buffer_monitor_thread = threading.Thread(
                target=self.monitor_audio_buffer, daemon=True
            )
            self.buffer_monitor_thread.start()

            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)

            # Start streaming
            logger.info("▶️  Starting Twitch stream...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.info("Pipeline starting asynchronously...")
                # Shorter timeout for Twitch
                ret, state, pending = self.pipeline.get_state(10 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.SUCCESS:
                    logger.info(
                        f"Pipeline started successfully, state: {state.value_nick}"
                    )
                elif ret == Gst.StateChangeReturn.ASYNC:
                    logger.warning(
                        "Pipeline still changing state, continuing anyway..."
                    )
                    # Force playing state
                    self.pipeline.set_state(Gst.State.PLAYING)
                else:
                    raise Exception(f"Pipeline failed to start: {ret}")
            elif ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("Pipeline started successfully")

            # Give pipeline time to stabilize
            time.sleep(2)

            # Generate initial audio
            initial_silence = bytes(
                int(AUDIO_SAMPLE_RATE * 0.5 * AUDIO_CHANNELS * 2)
            )  # 0.5 seconds
            self.audio_mixer.add_audio(initial_silence)
            logger.info("Added initial audio to start stream")

            # Start chat monitoring
            chat_thread = threading.Thread(target=self.monitor_twitch_chat, daemon=True)
            chat_thread.start()

            logger.info("\n" + "=" * 70)
            logger.info(
                "✅ ULTRA-LOW LATENCY TWITCH STREAMING WITH CLEAN AUDIO & SMOOTH WAVEFORM"
            )
            logger.info("📺 Check your Twitch channel")
            logger.info(f"🎬 Video: {BACKGROUND_VIDEO_PATH}")
            logger.info(f"🔊 Voice: {OPENAI_VOICE} (Realtime API)")
            logger.info("🌊 Smooth audio waveform visualization (no freezing)")
            logger.info("💬 Simulated chat responses")
            logger.info("🎵 Clean audio playback with noise reduction")
            logger.info("=" * 70 + "\n")

            # Main loop
            loop = GLib.MainLoop()
            loop.run()

        except Exception as e:
            logger.error(f"Twitch streaming error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop_streaming()

    def stop_streaming(self) -> None:
        """Stop streaming"""
        self.is_streaming = False
        self.waveform_running = False

        # Stop Realtime API
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Stop YouTube broadcast
        if self.platform == "youtube" and hasattr(self.youtube_api, "broadcast_id"):
            self.youtube_api.stop_broadcast()

        logger.info("✅ Stream stopped")

    def run(
        self, platform: str, broadcast_id: str = None, stream_key: str = None
    ) -> None:
        """Run the streamer"""
        try:
            print("\n" + "=" * 70)
            print("    🚀 ULTRA-FAST AI STREAMER WITH CLEAN AUDIO & SMOOTH WAVEFORM")
            print("    ⚡ OpenAI Realtime API")
            print("    🏎️  Safe & Fast chat monitoring")
            print("    🎵 Clean audio playback (no noise)")
            print("    🎬 Video background support")
            print("    🌊 Smooth waveform visualization (no freezing)")
            print("=" * 70)
            print(f"📺 Platform: {platform.upper()}")
            print(f"🎥 Video: {VIDEO_WIDTH}x{VIDEO_HEIGHT}@{VIDEO_FRAMERATE}fps")
            print(f"🔊 Audio: 24kHz Realtime ({OPENAI_VOICE})")
            print(f"📁 Background: {BACKGROUND_VIDEO_PATH}")
            print(
                f"🌊 Waveform: {WAVEFORM_WIDTH}x{WAVEFORM_HEIGHT} ({WAVEFORM_BARS} bars)"
            )
            print("=" * 70 + "\n")

            if platform == "youtube":
                self.start_youtube_streaming(use_existing_broadcast=broadcast_id)
            elif platform == "twitch":
                if not stream_key:
                    stream_key = input("Twitch stream key: ").strip()
                self.start_twitch_streaming(stream_key)
            else:
                logger.error(f"Unknown platform: {platform}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


# YouTube API Manager with Ultra-Fast methods
class YouTubeAPIManager:
    """YouTube API manager with ultra-fast chat fetching"""

    def __init__(self):
        self.youtube = None
        self.broadcast_id = None
        self.stream_id = None
        self.live_chat_id = None
        self.stream_key = None
        self._chat_request = None  # Reusable request object

    def authenticate(self) -> None:
        """Authenticate with YouTube"""
        try:
            creds = Credentials(
                None,
                refresh_token=YOUTUBE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=YOUTUBE_CLIENT_ID,
                client_secret=YOUTUBE_CLIENT_SECRET,
                scopes=YOUTUBE_SCOPES,
            )

            creds.refresh(Request())
            self.youtube = build("youtube", "v3", credentials=creds)
            logger.info("✅ YouTube authenticated")

        except Exception as e:
            logger.error(f"YouTube auth error: {e}")
            raise

    def create_broadcast(self, title: str = None, description: str = None) -> str:
        """Create YouTube broadcast"""
        try:
            if not title:
                title = f"🤖 AI Stream - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if not description:
                description = "AI stream with ultra-low latency voice responses and smooth audio waveform visualization!"

            broadcast_response = (
                self.youtube.liveBroadcasts()
                .insert(
                    part="snippet,status,contentDetails",
                    body={
                        "snippet": {
                            "title": title,
                            "description": description,
                            "scheduledStartTime": datetime.utcnow().isoformat() + "Z",
                            "enableLiveChat": True,
                        },
                        "status": {
                            "privacyStatus": "public",
                            "selfDeclaredMadeForKids": False,
                        },
                        "contentDetails": {
                            "enableAutoStart": True,
                            "enableAutoStop": True,
                            "latencyPreference": "ultraLow",
                            "enableDvr": False,
                            "projection": "rectangular",
                            "enableLiveChat": True,
                        },
                    },
                )
                .execute()
            )

            self.broadcast_id = broadcast_response["id"]
            logger.info(f"✅ Created broadcast: {self.broadcast_id}")

            return self.broadcast_id

        except HttpError as e:
            logger.error(f"Broadcast creation error: {e}")
            raise

    def create_stream(self) -> str:
        """Create stream"""
        try:
            stream_response = (
                self.youtube.liveStreams()
                .insert(
                    part="snippet,cdn,contentDetails,status",
                    body={
                        "snippet": {
                            "title": f"Stream {datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        },
                        "cdn": {
                            "frameRate": f"{VIDEO_FRAMERATE}fps",
                            "ingestionType": "rtmp",
                            "resolution": f"{VIDEO_HEIGHT}p",
                        },
                        "contentDetails": {"isReusable": False},
                    },
                )
                .execute()
            )

            self.stream_id = stream_response["id"]
            self.stream_key = stream_response["cdn"]["ingestionInfo"]["streamName"]

            logger.info(f"✅ Created stream: {self.stream_id}")
            logger.info(f"📝 Stream key: {self.stream_key[:10]}...")

            return self.stream_key

        except HttpError as e:
            logger.error(f"Stream creation error: {e}")
            raise

    def bind_broadcast_to_stream(self) -> None:
        """Bind broadcast to stream"""
        try:
            self.youtube.liveBroadcasts().bind(
                part="id,contentDetails", id=self.broadcast_id, streamId=self.stream_id
            ).execute()

            logger.info("✅ Broadcast bound to stream")

        except HttpError as e:
            logger.error(f"Binding error: {e}")
            raise

    def wait_for_stream_active(self, timeout: int = 60) -> bool:
        """Wait for stream to become active"""
        logger.info("⏳ Waiting for stream to become active...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                stream = (
                    self.youtube.liveStreams()
                    .list(part="status", id=self.stream_id)
                    .execute()
                )

                if stream["items"]:
                    status = stream["items"][0]["status"]["streamStatus"]
                    logger.info(f"Stream status: {status}")
                    if status == "active":
                        logger.info("✅ Stream is active")
                        return True

                time.sleep(2)

            except Exception as e:
                logger.error(f"Stream status check error: {e}")

        return False

    def transition_to_live(self) -> None:
        """Transition broadcast to live"""
        try:
            broadcast = (
                self.youtube.liveBroadcasts()
                .list(part="status,snippet", id=self.broadcast_id)
                .execute()
            )

            if broadcast["items"]:
                current_status = broadcast["items"][0]["status"]["lifeCycleStatus"]

                if current_status == "live":
                    logger.info("✅ Broadcast already live")
                    self.live_chat_id = broadcast["items"][0]["snippet"].get(
                        "liveChatId"
                    )
                    logger.info(f"📝 Live chat ID: {self.live_chat_id}")
                    return

                self.youtube.liveBroadcasts().transition(
                    broadcastStatus="live", id=self.broadcast_id, part="status"
                ).execute()

                logger.info("✅ Broadcast transitioned to live")

                # Wait a moment for the transition to complete
                time.sleep(2)

                # Get live chat ID
                broadcast = (
                    self.youtube.liveBroadcasts()
                    .list(part="snippet", id=self.broadcast_id)
                    .execute()
                )

                if broadcast["items"]:
                    self.live_chat_id = broadcast["items"][0]["snippet"].get(
                        "liveChatId"
                    )
                    logger.info(f"📝 Live chat ID: {self.live_chat_id}")

        except HttpError as e:
            if "redundantTransition" in str(e):
                logger.info("✅ Broadcast already live")
                try:
                    broadcast = (
                        self.youtube.liveBroadcasts()
                        .list(part="snippet", id=self.broadcast_id)
                        .execute()
                    )

                    if broadcast["items"]:
                        self.live_chat_id = broadcast["items"][0]["snippet"].get(
                            "liveChatId"
                        )
                        logger.info(f"📝 Live chat ID: {self.live_chat_id}")
                except Exception as fetch_error:
                    logger.error(f"Failed to fetch live chat ID: {fetch_error}")
            else:
                logger.error(f"Transition error: {e}")
                raise

    def get_chat_messages_fast(self, page_token: str = None) -> Dict[str, Any]:
        """Ultra-fast optimized chat message fetching"""
        if not self.live_chat_id:
            return {}

        try:
            params = {
                "liveChatId": self.live_chat_id,
                "part": "snippet,authorDetails",
                "maxResults": 2000,
                "fields": "items(snippet(type,displayMessage,publishedAt),authorDetails/displayName),nextPageToken,pollingIntervalMillis",
            }

            if page_token:
                params["pageToken"] = page_token

            if not self._chat_request:
                self._chat_request = self.youtube.liveChatMessages()

            response = self._chat_request.list(**params).execute()

            messages = [
                {
                    "author": item["authorDetails"]["displayName"],
                    "message": item["snippet"]["displayMessage"],
                    "timestamp": item["snippet"]["publishedAt"],
                }
                for item in response.get("items", [])
                if item["snippet"]["type"] == "textMessageEvent"
            ]

            return {
                "messages": messages,
                "nextPageToken": response.get("nextPageToken"),
                "pollingIntervalMillis": response.get("pollingIntervalMillis", 500),
            }

        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return {}

    def stop_broadcast(self) -> None:
        """Stop the broadcast"""
        try:
            if self.broadcast_id:
                self.youtube.liveBroadcasts().transition(
                    broadcastStatus="complete", id=self.broadcast_id, part="status"
                ).execute()
                logger.info("✅ Broadcast stopped")

        except Exception as e:
            logger.error(f"Broadcast stop error: {e}")


# Twitch Chat Simulator
class TwitchChatSimulator:
    """Simulates Twitch chat messages"""

    def __init__(self):
        self.message_pool = TWITCH_SIMULATED_MESSAGES.copy()
        self.used_messages = []
        self.last_message_time = time.time()
        self.min_interval = 15
        self.max_interval = 35

    def get_next_message(self) -> Optional[Dict[str, str]]:
        """Get next simulated message"""
        current_time = time.time()
        time_since_last = current_time - self.last_message_time

        next_interval = random.uniform(self.min_interval, self.max_interval)

        if time_since_last >= next_interval:
            if not self.message_pool:
                self.message_pool = self.used_messages.copy()
                self.used_messages = []
                random.shuffle(self.message_pool)

            if self.message_pool:
                message = self.message_pool.pop(
                    random.randint(0, len(self.message_pool) - 1)
                )
                self.used_messages.append(message)
                self.last_message_time = current_time
                message["timestamp"] = datetime.now().isoformat()
                return message

        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ultra-Fast AI Streamer with Clean Audio & Smooth Waveform"
    )
    parser.add_argument(
        "--platform",
        choices=["youtube", "twitch"],
        required=True,
        help="Platform to stream to",
    )
    parser.add_argument("--broadcast-id", help="Existing YouTube broadcast ID")
    parser.add_argument(
        "--stream-key", help="Stream key (for Twitch or manual YouTube)"
    )

    args = parser.parse_args()

    streamer = WorkingLiveAIStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main()
