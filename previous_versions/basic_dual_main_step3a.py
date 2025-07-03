#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 3a (Conservative)
Keep working Step 2 structure but add alternating robot voices
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


class ConservativeDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Step 3a: Conservative dual robot approach
    - Keep exact working Step 2 structure
    - Add voice alternation between robots
    - Simple but effective dual robot feel
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"  # Start with alloy
        self.robot_voices = ["alloy", "echo"]  # Two different voices
        self.conversation_count = 0
        logger.info("🤖🤖 ConservativeDualRobotStreamer initialized")
    
    def start_event_loop(self):
        """Override to use dual robot instructions with voice alternation"""
        logger.info("🤖🤖 Setting up Conservative Dual Robot API...")
        
        # Use the exact same threading setup as parent, but with dual robot instructions
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Create Realtime client with DUAL robot instructions
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 25 words.
        Have engaging back-and-forth conversations between the two robots!
        
        Example conversation:
        [Robot-L] Hey Robot-R! What do you think is the most amazing thing about streaming technology?
        [Robot-R] The real-time connection between minds across vast distances, Robot-L. What fascinates you most?
        [Robot-L] The instant feedback! It's like we're all in the same room together!
        """
        
        # Start with a random voice
        self.current_robot_voice = random.choice(self.robot_voices)
        logger.info(f"🔊 Starting with voice: {self.current_robot_voice}")
        
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            self.current_robot_voice,  # Use current robot voice
            dual_robot_instructions,
            self.audio_received_callback, 
            self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule voice changes and robot conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def _start_robot_conversation(self):
        """Start automatic robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Alternate robot personalities in messages
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello everyone! I'm excited to chat today. What's on your mind?"
            else:
                starter_message = "[Robot-R] Greetings! Let's explore some fascinating topics together."
            
            # Send the conversation starter
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule next conversation
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            conversation_topics = [
                "[Robot-L] Robot-R, what do you think about the future of AI streaming?",
                "[Robot-R] Robot-L, I find the interaction between technology and creativity quite intriguing.",
                "[Robot-L] The real-time nature of streaming fascinates me! What's your perspective, Robot-R?",
                "[Robot-R] Robot-L, consider how we bridge digital and human experiences through conversation.",
                "[Robot-L] I love how each chat message brings new perspectives! Don't you agree, Robot-R?",
                "[Robot-R] Indeed, Robot-L. The diversity of human thoughts creates endless learning opportunities."
            ]
            
            # Pick a random conversation topic
            topic = random.choice(conversation_topics)
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            
            # Alternate voice for next response
            self._switch_robot_voice()
            
            # Schedule next conversation (random interval)
            next_conversation = random.uniform(12.0, 20.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)
    
    def _switch_robot_voice(self):
        """Switch to the other robot voice"""
        current_index = self.robot_voices.index(self.current_robot_voice)
        next_index = (current_index + 1) % len(self.robot_voices)
        self.current_robot_voice = self.robot_voices[next_index]
        
        # Update the realtime client voice
        if self.realtime_client:
            # Note: Voice changes take effect on next connection
            # For now, just log the change
            logger.info(f"🔄 Switching to voice: {self.current_robot_voice}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Conservative Dual Robot AI Streamer - Step 3a"
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

    logger.info("🤖🤖 Starting Conservative Dual Robot AI Streamer - Step 3a...")
    logger.info("📝 This keeps Step 2 working structure but adds voice variety")
    logger.info("🎯 Goal: Reliable dual robot conversations with working audio")
    logger.info("🔊 Audio: Proven working pipeline from Step 2")

    # Use our conservative dual robot streamer
    streamer = ConservativeDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 