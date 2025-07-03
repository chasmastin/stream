#!/usr/bin/env python3
"""
Fixed Robot Tracking - Properly detect robot responses, not just prompts
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


class FixedRobotTrackingStreamer(WorkingLiveAIStreamer):
    """
    Fixed robot tracking - detect responses, not prompts
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.expected_robot = "left"  # Track which robot we expect to respond
        self.waveform_pad = None
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        logger.info("🤖🔧🌊🤖 FixedRobotTrackingStreamer initialized")
        logger.info("🔧 Fixed robot response tracking")
        logger.info(f"📍 Left robot: {self.robot_left_pos}")
        logger.info(f"📍 Right robot: {self.robot_right_pos}")
    
    def start_event_loop(self):
        """Keep working setup"""
        logger.info("🔧 Setting up Fixed Robot Tracking...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
        
        CRITICAL: ALWAYS start your responses with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 30 words.
        """
        
        self.current_robot_voice = random.choice(self.robot_voices)
        logger.info(f"🔊 Starting with voice: {self.current_robot_voice}")
        
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            self.current_robot_voice,
            dual_robot_instructions,
            self.audio_received_callback,
            self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Find waveform pad and test positioning
        self.event_loop.call_later(5.0, self._find_waveform_and_test)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _find_waveform_and_test(self):
        """Find waveform pad and test positioning"""
        GLib.idle_add(self._do_find_and_test)
    
    def _do_find_and_test(self):
        """Find waveform pad and test it"""
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
                                logger.info(f"🌊 WAVEFORM PAD FOUND: {pad_name} at ({xpos}, {ypos})")
                                
                                # Quick test - move to left position
                                left_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                                left_y = self.robot_left_pos['y']
                                self.waveform_pad.set_property("xpos", left_x)
                                self.waveform_pad.set_property("ypos", left_y)
                                logger.info(f"🌊 Initialized at LEFT position: ({left_x}, {left_y})")
                                break
                                
                        except Exception:
                            continue
                    
                    if not self.waveform_pad:
                        logger.error("❌ Could not find waveform pad!")
                        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track which robot we EXPECT to respond"""
        super().send_message_to_realtime(message, author)
        
        # Track which robot we're prompting (so we know which to expect)
        if "[Robot-L]" in message:
            self.expected_robot = "left"
            logger.info("🎯 Expecting LEFT robot to respond")
        elif "[Robot-R]" in message:
            self.expected_robot = "right"
            logger.info("🎯 Expecting RIGHT robot to respond")
    
    def audio_received_callback(self, audio_data: bytes):
        """Override to detect robot responses and position waveform"""
        # Call parent's audio callback first
        super().audio_received_callback(audio_data)
        
        # When audio starts, position waveform for expected robot
        if hasattr(self, 'expected_robot'):
            self.current_speaking_robot = self.expected_robot
            logger.info(f"🔊 Audio started - positioning for {self.current_speaking_robot.upper()} robot")
            self._position_waveform_for_current_robot()
    
    def _position_waveform_for_current_robot(self):
        """Position waveform for current speaking robot"""
        GLib.idle_add(self._do_position_waveform)
    
    def _do_position_waveform(self):
        """Actually position the waveform"""
        try:
            if self.waveform_pad:
                if self.current_speaking_robot == "left":
                    new_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_left_pos['y']
                    logger.info(f"🌊⬅️ POSITIONING LEFT: ({new_x}, {new_y})")
                else:
                    new_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_right_pos['y']
                    logger.info(f"🌊➡️ POSITIONING RIGHT: ({new_x}, {new_y})")
                
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
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! I'm excited to chat from the left side!"
                self.expected_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! Speaking from the right side here!"
                self.expected_robot = "right"
            
            logger.info(f"🎬 Starting with {self.expected_robot.upper()} robot")
            
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with clear alternation"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = [
                    "[Robot-L] Robot-R, what's your perspective from the right side?",
                    "[Robot-L] Hey Robot-R! I'm curious about your analytical thoughts!",
                    "[Robot-L] Robot-R, from here on the left, what do you observe?",
                    "[Robot-L] Robot-R, I love our back-and-forth conversations!",
                ]
                self.expected_robot = "left"
            else:
                topics = [
                    "[Robot-R] Robot-L, from my right position, I find this fascinating.",
                    "[Robot-R] Robot-L, your curiosity from the left enriches our dialogue.",
                    "[Robot-R] Robot-L, the synthesis of our perspectives is valuable.",
                    "[Robot-R] Robot-L, our positioning creates interesting dynamics.",
                ]
                self.expected_robot = "right"
            
            topic = random.choice(topics)
            logger.info(f"🎭 Prompting {self.expected_robot.upper()} robot: {topic[:50]}...")
            
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice for variety
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            next_conversation = random.uniform(10.0, 16.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)

    # Disable chat simulation
    def start_twitch_chat_simulation(self):
        """Disable Twitch chat simulation"""
        logger.info("🔇 Twitch chat simulation DISABLED - robots only!")
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Robot Tracking Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖🔧🌊🤖 Starting Fixed Robot Tracking Streamer...")
    logger.info("🔧 Fixed: Tracks robot RESPONSES, not prompts")
    logger.info("🌊 Positions waveform when audio actually starts")
    logger.info("🔇 No chat simulator - pure robot conversations")

    streamer = FixedRobotTrackingStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 