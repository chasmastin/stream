#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 3a + Positioned Waveforms
Keep exact Step 3a working audio but add positioned waveforms
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import asyncio
import random
from collections import deque
from datetime import datetime

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib, GObject, GstVideo

# Initialize GStreamer FIRST
Gst.init(None)

# Import our modules
from modules import config
from modules.audio import AudioWaveformGenerator, SimpleAudioMixer
from modules.api import SimpleRealtimeAPIClient
from modules.platforms import YouTubeAPIManager, TwitchChatSimulator
from modules.streaming.streamer import WorkingLiveAIStreamer

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable debug logging for waveform tracking
logging.getLogger("modules.streaming.streamer").setLevel(logging.DEBUG)


class PositionedWaveformDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Step 3a + Positioned Waveforms
    - Keep EXACT Step 3a audio pipeline (don't touch audio callbacks)
    - Add positioned waveforms based on which robot is speaking
    - Track robot speaking state and update waveform position
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"  # Track which robot is speaking
        self.waveform_position_updated = False
        
        logger.info("🤖🌊🤖 PositionedWaveformDualRobotStreamer initialized")
        logger.info("🔊 Using EXACT Step 3a audio pipeline")
        logger.info("🌊 Adding positioned waveforms based on speaking robot")
    
    def start_event_loop(self):
        """Keep exact Step 3a event loop"""
        logger.info("🤖🤖 Setting up Positioned Waveform Dual Robot API...")
        
        # Use the exact same threading setup as Step 3a
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Enhanced dual robot instructions with position references
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
                          You're positioned on the LEFT side of the screen.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
                           You're positioned on the RIGHT side of the screen.
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 25 words.
        Have engaging back-and-forth conversations, occasionally referencing your positions!
        
        Example conversation:
        [Robot-L] Hey Robot-R! From the left side here, what's your take on AI streaming?
        [Robot-R] Robot-L, from my right-side perspective, I find the real-time interaction fascinating.
        [Robot-L] I love how we complement each other across the screen!
        """
        
        # Start with a random voice
        self.current_robot_voice = random.choice(self.robot_voices)
        logger.info(f"🔊 Starting with voice: {self.current_robot_voice}")
        
        # Use EXACT same Realtime client setup as Step 3a
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            self.current_robot_voice,
            dual_robot_instructions,
            self.audio_received_callback,  # Keep parent's callback
            self.audio_level_callback      # Keep parent's callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule robot conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track which robot is speaking based on message content"""
        # Call parent method first
        super().send_message_to_realtime(message, author)
        
        # Track which robot is about to speak based on message content
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot will respond - waveform will be on LEFT")
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot will respond - waveform will be on RIGHT")
        
        # Update waveform position for next response
        self._update_waveform_position()
    
    def _update_waveform_position(self):
        """Update waveform position based on current speaking robot"""
        if not hasattr(self, 'waveform_overlay') or not self.waveform_overlay:
            return
        
        # Get the waveform compositor pad (if it exists)
        try:
            # The waveform is connected to compositor pad sink_1 (overlay pad)
            if hasattr(self, 'pipeline') and self.pipeline:
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    waveform_pad = compositor.get_static_pad("sink_1")
                    if waveform_pad:
                        if self.current_speaking_robot == "left":
                            # Position waveform on left side
                            waveform_x = config.ROBOT_POSITIONS['left']['x']
                            waveform_y = config.ROBOT_POSITIONS['left']['y']
                            logger.info(f"🌊⬅️ Positioning waveform on LEFT: ({waveform_x}, {waveform_y})")
                        else:
                            # Position waveform on right side
                            waveform_x = config.ROBOT_POSITIONS['right']['x']
                            waveform_y = config.ROBOT_POSITIONS['right']['y']
                            logger.info(f"🌊➡️ Positioning waveform on RIGHT: ({waveform_x}, {waveform_y})")
                        
                        # Update the pad position
                        waveform_pad.set_property("xpos", waveform_x)
                        waveform_pad.set_property("ypos", waveform_y)
                        
                        logger.debug(f"✅ Waveform positioned at ({waveform_x}, {waveform_y}) for {self.current_speaking_robot} robot")
        except Exception as e:
            logger.debug(f"Waveform positioning update failed (this is normal during startup): {e}")
    
    def _start_robot_conversation(self):
        """Enhanced robot conversations with position awareness"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Alternate robot personalities in messages
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello from the left side! I'm excited to chat today!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings from the right side! Ready for fascinating discussions."
                self.current_speaking_robot = "right"
            
            logger.info(f"🤖 {self.current_speaking_robot.upper()} robot starting conversation")
            
            # Send the conversation starter
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule next conversation
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with position tracking"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Determine which robot is speaking
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, from the left side here, what's your view on AI streaming?",
                    "[Robot-L] Hey right-side buddy! The real-time connections fascinate me!",
                    "[Robot-L] Robot-R, I love getting different perspectives from your side of the screen!",
                    "[Robot-L] From over here on the left, I see endless possibilities!",
                    "[Robot-L] Robot-R, our dual perspective creates such rich conversations!"
                ]
                self.current_speaking_robot = "left"
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, from my position on the right, I find human-AI interaction intriguing.",
                    "[Robot-R] Left-side friend, our different viewpoints create meaningful dialogue.",
                    "[Robot-R] Robot-L, the synthesis of our perspectives enriches every conversation.",
                    "[Robot-R] From the right side here, I observe how technology bridges distances.",
                    "[Robot-R] Robot-L, our complementary positions symbolize balanced discourse."
                ]
                self.current_speaking_robot = "right"
            
            # Pick a random conversation topic for the current robot
            topic = random.choice(conversation_topics)
            logger.info(f"🗣️ {self.current_speaking_robot.upper()} robot speaking: {topic[:50]}...")
            
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Alternate voice for next response
            self._switch_robot_voice()
            
            # Schedule next conversation (random interval)
            next_conversation = random.uniform(12.0, 18.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)
    
    def _switch_robot_voice(self):
        """Switch to the other robot voice"""
        current_index = self.robot_voices.index(self.current_robot_voice)
        next_index = (current_index + 1) % len(self.robot_voices)
        self.current_robot_voice = self.robot_voices[next_index]
        
        logger.info(f"🔄 Next robot will use voice: {self.current_robot_voice}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Positioned Waveform Dual Robot AI Streamer - Step 3a Enhanced"
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

    logger.info("🤖🌊🤖 Starting Positioned Waveform Dual Robot AI Streamer...")
    logger.info("📝 This adds positioned waveforms to Step 3a working system")
    logger.info("🎯 Goal: Waveforms positioned based on which robot is speaking")
    logger.info("🔊 Audio: EXACT same reliable pipeline from Step 3a")
    logger.info("🌊 Enhancement: Dynamic waveform positioning (left/right)")

    # Use our positioned waveform dual robot streamer
    streamer = PositionedWaveformDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 