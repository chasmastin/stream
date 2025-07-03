#!/usr/bin/env python3
"""
Dual Voice Robots - Two separate API connections
- Left Robot: Male voice (echo) - separate API connection
- Right Robot: Female voice (nova) - separate API connection
- Positioned waveforms
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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DualVoiceRobotStreamer(WorkingLiveAIStreamer):
    """
    Two separate API connections for different voices
    """
    
    def __init__(self):
        super().__init__()
        
        # Two separate API clients
        self.left_robot_client = None   # Male voice (echo)
        self.right_robot_client = None  # Female voice (nova)
        
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        logger.info("🤖👨👩🌊 DualVoiceRobotStreamer initialized")
        logger.info("👨 Left Robot: MALE voice (echo) - separate API")
        logger.info("👩 Right Robot: FEMALE voice (nova) - separate API")
        logger.info("🌊 Positioned waveforms")
    
    def start_event_loop(self):
        """Setup with two separate API connections"""
        logger.info("👨👩 Setting up Dual Voice Robot APIs...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Distinct instructions for each robot
        left_robot_instructions = """You are Robot-L, a curious and enthusiastic male robot positioned on the left side. You love to start conversations, ask questions, and explore ideas with excitement. Always start your responses with [Robot-L]. Keep responses under 30 words. You're having a conversation with Robot-R, your analytical female counterpart on the right."""

        right_robot_instructions = """You are Robot-R, a wise and analytical female robot positioned on the right side. You provide thoughtful insights, build on ideas, and offer reflective perspectives. Always start your responses with [Robot-R]. Keep responses under 30 words. You're having a conversation with Robot-L, your curious male counterpart on the left."""
        
        # Create two separate API clients
        logger.info("🔄 Creating LEFT robot API (male voice)...")
        self.left_robot_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            "echo",  # Male voice
            left_robot_instructions,
            self.audio_received_callback,
            self.audio_level_callback
        )
        
        logger.info("🔄 Creating RIGHT robot API (female voice)...")
        self.right_robot_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            "nova",  # Female voice
            right_robot_instructions,
            self.audio_received_callback,
            self.audio_level_callback
        )
        
        # Connect both clients
        self.event_loop.run_until_complete(self._connect_both_clients())

        # Find waveform pad and start conversations
        self.event_loop.call_later(5.0, self._find_waveform_and_test)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    async def _connect_both_clients(self):
        """Connect both API clients"""
        try:
            logger.info("🔌 Connecting LEFT robot (male)...")
            await self.left_robot_client.connect()
            logger.info("✅ LEFT robot connected!")
            
            logger.info("🔌 Connecting RIGHT robot (female)...")
            await self.right_robot_client.connect()
            logger.info("✅ RIGHT robot connected!")
            
            logger.info("🎉 Both robot voices ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect robots: {e}")
    
    def _find_waveform_and_test(self):
        """Find waveform pad"""
        GLib.idle_add(self._do_find_waveform)
    
    def _do_find_waveform(self):
        """Find waveform pad"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Finding waveform pad...")
                    
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        try:
                            xpos = pad.get_property("xpos")
                            ypos = pad.get_property("ypos")
                            
                            if "sink_1" in pad_name:
                                self.waveform_pad = pad
                                logger.info(f"🌊 WAVEFORM PAD FOUND: {pad_name}")
                                
                                # Initialize at left position
                                left_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                                left_y = self.robot_left_pos['y']
                                self.waveform_pad.set_property("xpos", left_x)
                                self.waveform_pad.set_property("ypos", left_y)
                                logger.info(f"🌊 Initialized at LEFT position: ({left_x}, {left_y})")
                                break
                                
                        except Exception:
                            continue
                        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Route messages to appropriate robot API"""
        # Determine which robot should respond
        if "[Robot-L]" in message or self.current_speaking_robot == "left":
            if self.left_robot_client and self.left_robot_client.is_connected:
                logger.info("📤 Sending to LEFT robot (male voice)")
                self.current_speaking_robot = "left"
                self._position_waveform_for_robot("left")
                asyncio.create_task(self.left_robot_client.send_text_message(message, author))
            
        elif "[Robot-R]" in message or self.current_speaking_robot == "right":
            if self.right_robot_client and self.right_robot_client.is_connected:
                logger.info("📤 Sending to RIGHT robot (female voice)")
                self.current_speaking_robot = "right"
                self._position_waveform_for_robot("right")
                asyncio.create_task(self.right_robot_client.send_text_message(message, author))
        
        else:
            # Default to left robot
            if self.left_robot_client and self.left_robot_client.is_connected:
                logger.info("📤 Defaulting to LEFT robot")
                self.current_speaking_robot = "left"
                self._position_waveform_for_robot("left")
                asyncio.create_task(self.left_robot_client.send_text_message(message, author))
    
    def _position_waveform_for_robot(self, robot: str):
        """Position waveform for specified robot"""
        self.current_speaking_robot = robot
        GLib.idle_add(self._do_position_waveform)
    
    def _do_position_waveform(self):
        """Actually position the waveform"""
        try:
            if self.waveform_pad:
                if self.current_speaking_robot == "left":
                    new_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_left_pos['y']
                    logger.info(f"🌊👨 POSITIONING LEFT (MALE): ({new_x}, {new_y})")
                else:
                    new_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_right_pos['y']
                    logger.info(f"🌊👩 POSITIONING RIGHT (FEMALE): ({new_x}, {new_y})")
                
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform positioned at: ({actual_x}, {actual_y})")
                
        except Exception as e:
            logger.error(f"❌ Positioning failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.left_robot_client and self.left_robot_client.is_connected:
            starter_message = "Hello! I'm excited to start our conversation!"
            logger.info("🎬 Starting with LEFT robot (male)")
            self.current_speaking_robot = "left"
            self._position_waveform_for_robot("left")
            asyncio.create_task(self.left_robot_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations alternating between APIs"""
        is_left_turn = (self.conversation_count % 2 == 0)
        
        if is_left_turn and self.left_robot_client and self.left_robot_client.is_connected:
            # Left robot (male) turn
            topics = [
                "What's your perspective on AI technology?",
                "I'm curious about your thoughts on streaming!",
                "What excites you most about our conversations?",
                "How do you see the future of AI?",
            ]
            topic = random.choice(topics)
            logger.info(f"🎭 LEFT robot (MALE): {topic}")
            self.current_speaking_robot = "left"
            self._position_waveform_for_robot("left")
            asyncio.create_task(self.left_robot_client.send_text_message(topic, "RobotConversation"))
            
        elif not is_left_turn and self.right_robot_client and self.right_robot_client.is_connected:
            # Right robot (female) turn
            topics = [
                "From my analytical perspective, I find human-AI interaction fascinating.",
                "Your enthusiasm perfectly complements my thoughtful approach.",
                "I've been considering the patterns in our dialogue.",
                "The synthesis of different viewpoints creates rich conversation.",
            ]
            topic = random.choice(topics)
            logger.info(f"🎭 RIGHT robot (FEMALE): {topic}")
            self.current_speaking_robot = "right"
            self._position_waveform_for_robot("right")
            asyncio.create_task(self.right_robot_client.send_text_message(topic, "RobotConversation"))
        
        self.conversation_count += 1
        next_conversation = random.uniform(10.0, 16.0)
        self.event_loop.call_later(next_conversation, self._continue_robot_conversation)

    # Disable chat simulation
    def monitor_twitch_chat(self):
        """Disable Twitch chat simulation completely"""
        logger.info("🔇 Twitch chat simulation DISABLED - robots only!")
        pass

    # Override parent's initialize_realtime_api to prevent single API setup
    def initialize_realtime_api(self):
        """Skip single API initialization - we use dual APIs"""
        logger.info("🔄 Skipping single API setup - using dual robot APIs")
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dual Voice Robot Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖👨👩🌊 Starting Dual Voice Robot Streamer...")
    logger.info("👨 Left Robot: MALE voice (echo) - separate API connection")
    logger.info("👩 Right Robot: FEMALE voice (nova) - separate API connection")
    logger.info("🌊 Positioned waveforms based on speaking robot")
    logger.info("🔇 No chat simulator - pure robot conversations")

    streamer = DualVoiceRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 