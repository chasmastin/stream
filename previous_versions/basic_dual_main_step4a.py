#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 4a (Conservative)
Keep exact Step 3a audio pipeline but add better robot tracking
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


class ConservativePositionedDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Step 4a: Conservative positioned dual robot approach
    - Keep EXACT Step 3a audio pipeline (don't touch audio_received_callback)
    - Add better robot tracking and identification
    - Enhanced conversation flow
    - Same reliable audio routing
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"  # Just for tracking, don't change audio
        
        logger.info("🤖📍🤖 ConservativePositionedDualRobotStreamer initialized")
        logger.info("🔊 Using EXACT Step 3a audio pipeline - no changes")
        logger.info("📍 Adding robot position tracking for future enhancements")
    
    def start_event_loop(self):
        """Keep exact Step 3a event loop - don't change anything audio-related"""
        logger.info("🤖🤖 Setting up Conservative Positioned Dual Robot API...")
        
        # Use the exact same threading setup as Step 3a
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Enhanced dual robot instructions with better positioning
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
                          You're positioned on the LEFT side of the screen.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
                           You're positioned on the RIGHT side of the screen.
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 30 words.
        Have engaging back-and-forth conversations, referencing your positions occasionally!
        
        Example conversation:
        [Robot-L] Hey Robot-R! From my spot on the left, I can see you over there on the right!
        [Robot-R] Indeed, Robot-L! Our different perspectives create a wonderful dialogue.
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
            self.audio_received_callback,  # Use parent's callback - DON'T OVERRIDE
            self.audio_level_callback      # Use parent's callback - DON'T OVERRIDE
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule robot conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def _start_robot_conversation(self):
        """Enhanced robot conversations with position awareness"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Alternate robot personalities in messages
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello everyone! I'm Robot-L on the left side, excited to chat!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm Robot-R on the right side, ready for deep discussions."
                self.current_speaking_robot = "right"
            
            logger.info(f"🤖 {self.current_speaking_robot.upper()} robot starting conversation")
            
            # Send the conversation starter
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule next conversation
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Enhanced robot conversations with better positioning"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Determine which robot is speaking
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, from the left side here, what's your view on AI streaming?",
                    "[Robot-L] Hey right-side buddy! The real-time connections fascinate me!",
                    "[Robot-L] Robot-R, I love getting different perspectives from your side of the screen!",
                    "[Robot-L] From over here on the left, I see endless possibilities in AI communication!",
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
        description="Conservative Positioned Dual Robot AI Streamer - Step 4a"
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

    logger.info("🤖📍🤖 Starting Conservative Positioned Dual Robot AI Streamer - Step 4a...")
    logger.info("📝 This keeps Step 3a working audio but adds better robot tracking")
    logger.info("🎯 Goal: Enhanced robot conversations with position awareness")
    logger.info("🔊 Audio: EXACT same reliable pipeline from Step 3a")
    logger.info("📍 Enhancement: Better robot identification and conversations")

    # Use our conservative positioned dual robot streamer
    streamer = ConservativePositionedDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 