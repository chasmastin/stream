#!/usr/bin/env python3
"""
Step 3a + Enhanced Robot Tracking
Keep exact Step 3a working system but add better robot identification
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


class EnhancedRobotTrackingStreamer(WorkingLiveAIStreamer):
    """
    Step 3a + Enhanced Robot Tracking
    - Keep EXACT Step 3a working system
    - Add better robot identification and logging
    - Visual indicators for which robot is speaking
    - No waveform positioning changes (keep it simple and working)
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        
        logger.info("🤖👥🤖 EnhancedRobotTrackingStreamer initialized")
        logger.info("🔊 Using EXACT Step 3a audio pipeline")
        logger.info("👥 Adding enhanced robot tracking and identification")
    
    def start_event_loop(self):
        """Keep exact Step 3a event loop"""
        logger.info("🤖🤖 Setting up Enhanced Robot Tracking API...")
        
        # Use the exact same threading setup as Step 3a
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Enhanced dual robot instructions with clear positioning
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (LEFT SIDE): Curious, optimistic, loves to start conversations and ask questions.
                                Position: LEFT side of the screen at (320, 200).
        🤖 Robot-R (RIGHT SIDE): Analytical, wise, provides insights and builds on ideas.
                                 Position: RIGHT side of the screen at (960, 200).
        
        IMPORTANT: 
        - Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        - Alternate between robots naturally in conversation. 
        - Keep responses under 25 words.
        - Occasionally reference your position (left side vs right side).
        - Have engaging back-and-forth conversations!
        
        Example conversation:
        [Robot-L] Hey Robot-R! From here on the left, I'm excited about this stream!
        [Robot-R] Robot-L, from my right-side perspective, I find our dialogue fascinating.
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
            self.audio_received_callback,  # Keep parent's callback unchanged
            self.audio_level_callback      # Keep parent's callback unchanged
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule robot conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track which robot is speaking"""
        # Call parent method first (don't change functionality)
        super().send_message_to_realtime(message, author)
        
        # Enhanced robot tracking with visual indicators
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("=" * 50)
            logger.info("🤖⬅️  LEFT ROBOT SPEAKING  ⬅️🤖")
            logger.info(f"📍 Position: {config.ROBOT_POSITIONS['left']}")
            logger.info(f"🗣️ Message: {message[:60]}...")
            logger.info("🌊 Waveform should be on LEFT side")
            logger.info("=" * 50)
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("=" * 50)
            logger.info("🤖➡️  RIGHT ROBOT SPEAKING  ➡️🤖")
            logger.info(f"📍 Position: {config.ROBOT_POSITIONS['right']}")
            logger.info(f"🗣️ Message: {message[:60]}...")
            logger.info("🌊 Waveform should be on RIGHT side")
            logger.info("=" * 50)
    
    def _start_robot_conversation(self):
        """Enhanced robot conversations with clear positioning"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Alternate robot personalities in messages
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello everyone! I'm the LEFT robot, excited to chat from this side!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the RIGHT robot, ready for deep discussions from my side!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🎬 {self.current_speaking_robot.upper()} robot starting conversation")
            
            # Send the conversation starter
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule next conversation
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with enhanced positioning awareness"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Determine which robot is speaking
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, from the LEFT side here, what's your view on AI streaming?",
                    "[Robot-L] Hey RIGHT-side buddy! The real-time connections fascinate me!",
                    "[Robot-L] Robot-R, I love getting different perspectives from your RIGHT side!",
                    "[Robot-L] From over here on the LEFT, I see endless possibilities!",
                    "[Robot-L] Robot-R, our LEFT-RIGHT perspective creates such rich conversations!",
                    "[Robot-L] Speaking from the LEFT position, what do you think, Robot-R?"
                ]
                self.current_speaking_robot = "left"
                logger.info("📋 LEFT robot's turn to speak")
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, from my RIGHT position, I find human-AI interaction intriguing.",
                    "[Robot-R] LEFT-side friend, our different viewpoints create meaningful dialogue.",
                    "[Robot-R] Robot-L, the synthesis of our LEFT-RIGHT perspectives enriches conversation.",
                    "[Robot-R] From the RIGHT side here, I observe how technology bridges distances.",
                    "[Robot-R] Robot-L, our complementary LEFT-RIGHT positions symbolize balanced discourse.",
                    "[Robot-R] Speaking from the RIGHT position, Robot-L, what fascinates you?"
                ]
                self.current_speaking_robot = "right"
                logger.info("📋 RIGHT robot's turn to speak")
            
            # Pick a random conversation topic for the current robot
            topic = random.choice(conversation_topics)
            logger.info(f"🎭 {self.current_speaking_robot.upper()} robot preparing: {topic[:40]}...")
            
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
        description="Enhanced Robot Tracking Dual Robot AI Streamer"
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

    logger.info("🤖👥🤖 Starting Enhanced Robot Tracking Dual Robot AI Streamer...")
    logger.info("📝 This keeps Step 3a working audio and adds enhanced robot tracking")
    logger.info("🎯 Goal: Clear visual indicators of which robot is speaking")
    logger.info("🔊 Audio: EXACT same reliable pipeline from Step 3a")
    logger.info("👥 Enhancement: Better robot identification and logging")

    # Use our enhanced robot tracking streamer
    streamer = EnhancedRobotTrackingStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 