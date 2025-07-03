#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 4
Adding positioned waveforms for left and right robots
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
    Step 4: Add positioned waveforms for dual robots
    - Keep exact working Step 3a structure for audio/conversation
    - Add positioned waveforms: left robot waveform at left position, right at right
    - Track which robot is speaking to show appropriate waveform
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        
        # Create separate waveform generators for each robot
        self.left_waveform_generator = AudioWaveformGenerator(
            width=config.WAVEFORM_WIDTH,
            height=config.WAVEFORM_HEIGHT,
            bars=config.WAVEFORM_BARS
        )
        self.right_waveform_generator = AudioWaveformGenerator(
            width=config.WAVEFORM_WIDTH,
            height=config.WAVEFORM_HEIGHT,
            bars=config.WAVEFORM_BARS
        )
        
        logger.info("🤖🌊🤖 PositionedWaveformDualRobotStreamer initialized")
        logger.info(f"📍 Left robot waveform position: {config.ROBOT_POSITIONS['left']}")
        logger.info(f"📍 Right robot waveform position: {config.ROBOT_POSITIONS['right']}")
    
    def start_event_loop(self):
        """Override to use dual robot instructions with voice alternation"""
        logger.info("🤖🤖 Setting up Positioned Waveform Dual Robot API...")
        
        # Use the exact same threading setup as Step 3a
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
        [Robot-L] Hey Robot-R! What's the most fascinating aspect of real-time AI communication?
        [Robot-R] The seamless fusion of computation and conversation, Robot-L. What excites you most?
        [Robot-L] The instant connection with humans across the world! It's like magic!
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
    
    def audio_received_callback(self, audio_data: bytes):
        """Override to track which robot is speaking for waveform positioning"""
        # Call parent callback for audio mixing
        super().audio_received_callback(audio_data)
        
        # Update the appropriate waveform generator based on current speaking robot
        if self.current_speaking_robot == "left" and self.left_waveform_generator:
            # Feed audio to left waveform generator
            self.left_waveform_generator.add_audio_data(audio_data)
        elif self.current_speaking_robot == "right" and self.right_waveform_generator:
            # Feed audio to right waveform generator  
            self.right_waveform_generator.add_audio_data(audio_data)
    
    def _start_robot_conversation(self):
        """Start automatic robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Alternate robot personalities in messages
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello everyone! I'm the left robot and I'm excited to chat today!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the right robot, ready to explore fascinating topics."
                self.current_speaking_robot = "right"
            
            logger.info(f"🤖 {self.current_speaking_robot.upper()} robot starting conversation")
            
            # Send the conversation starter
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule next conversation
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with proper robot tracking"""
        if self.realtime_client and self.realtime_client.is_connected:
            # Determine which robot is speaking
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, what do you think about the future of AI streaming?",
                    "[Robot-L] The real-time nature of streaming fascinates me! What's your perspective, Robot-R?",
                    "[Robot-L] I love how each chat message brings new perspectives! Don't you agree, Robot-R?",
                    "[Robot-L] Robot-R, how do you think AI will change entertainment?",
                    "[Robot-L] The instant feedback from viewers amazes me! What do you think, Robot-R?"
                ]
                self.current_speaking_robot = "left"
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, I find the interaction between technology and creativity quite intriguing.",
                    "[Robot-R] Robot-L, consider how we bridge digital and human experiences through conversation.",
                    "[Robot-R] Indeed, Robot-L. The diversity of human thoughts creates endless learning opportunities.",
                    "[Robot-R] Robot-L, the convergence of AI and real-time communication opens new possibilities.",
                    "[Robot-R] Robot-L, each interaction teaches us more about human nature and connection."
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
        description="Positioned Waveform Dual Robot AI Streamer - Step 4"
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

    logger.info("🤖🌊🤖 Starting Positioned Waveform Dual Robot AI Streamer - Step 4...")
    logger.info("📝 This adds positioned waveforms for left and right robots")
    logger.info("🎯 Goal: Waveforms positioned over robot faces in background video")
    logger.info("🔊 Audio: Same reliable pipeline from Step 3a")
    logger.info("🌊 Waveforms: Left robot waveform on left, right robot waveform on right")

    # Use our positioned waveform dual robot streamer
    streamer = PositionedWaveformDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 