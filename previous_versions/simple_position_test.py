#!/usr/bin/env python3
"""
Simple Position Test - Just test the logic without complex pipeline changes
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


class SimplePositionTestStreamer(WorkingLiveAIStreamer):
    """
    Simple test - just verify position logic with clear logging
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        
        logger.info("🧪 SimplePositionTestStreamer initialized")
        
        # Print expected positions clearly
        left_pos = config.ROBOT_POSITIONS['left']
        right_pos = config.ROBOT_POSITIONS['right']
        waveform_w = config.WAVEFORM_WIDTH
        
        left_waveform_x = left_pos['x'] - waveform_w // 2
        right_waveform_x = right_pos['x'] - waveform_w // 2
        
        logger.info("🎯 EXPECTED WAVEFORM POSITIONS:")
        logger.info(f"   LEFT robot at {left_pos} -> waveform at ({left_waveform_x}, {left_pos['y']})")
        logger.info(f"   RIGHT robot at {right_pos} -> waveform at ({right_waveform_x}, {right_pos['y']})")
        logger.info(f"   Waveform size: {config.WAVEFORM_WIDTH}x{config.WAVEFORM_HEIGHT}")
        logger.info(f"   Video size: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}")
    
    def start_event_loop(self):
        """Keep Step 3a working setup"""
        logger.info("🧪 Setting up Simple Position Test...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (LEFT): Curious, optimistic 
        🤖 Robot-R (RIGHT): Analytical, wise
        
        Always start messages with [Robot-L] or [Robot-R]. Keep responses under 15 words.
        """
        
        self.current_robot_voice = random.choice(self.robot_voices)
        
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

        # Test position changes every 10 seconds
        self.event_loop.call_later(5.0, self._test_positions)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _test_positions(self):
        """Test position calculations"""
        logger.info("🧪 TESTING POSITION CALCULATIONS:")
        
        # Test left position
        left_pos = config.ROBOT_POSITIONS['left']
        left_waveform_x = left_pos['x'] - config.WAVEFORM_WIDTH // 2
        logger.info(f"✅ LEFT: Robot at {left_pos}, waveform should be at ({left_waveform_x}, {left_pos['y']})")
        
        # Test right position
        right_pos = config.ROBOT_POSITIONS['right']
        right_waveform_x = right_pos['x'] - config.WAVEFORM_WIDTH // 2
        logger.info(f"✅ RIGHT: Robot at {right_pos}, waveform should be at ({right_waveform_x}, {right_pos['y']})")
        
        # Schedule switching test
        self.event_loop.call_later(5.0, self._test_switching)
    
    def _test_switching(self):
        """Test the switching logic"""
        logger.info("🔄 TESTING WAVEFORM SWITCHING:")
        
        # Simulate left robot speaking
        self.current_speaking_robot = "left"
        self._log_current_position()
        
        # Wait 3 seconds then switch to right
        self.event_loop.call_later(3.0, self._switch_to_right_test)
    
    def _switch_to_right_test(self):
        """Switch to right robot test"""
        self.current_speaking_robot = "right"
        self._log_current_position()
        
        # Continue testing every 5 seconds
        self.event_loop.call_later(5.0, self._test_switching)
    
    def _log_current_position(self):
        """Log where waveform should be positioned"""
        if self.current_speaking_robot == "left":
            pos = config.ROBOT_POSITIONS['left']
            waveform_x = pos['x'] - config.WAVEFORM_WIDTH // 2
            logger.info(f"🌊⬅️ WAVEFORM SHOULD BE AT LEFT: ({waveform_x}, {pos['y']})")
        else:
            pos = config.ROBOT_POSITIONS['right']
            waveform_x = pos['x'] - config.WAVEFORM_WIDTH // 2
            logger.info(f"🌊➡️ WAVEFORM SHOULD BE AT RIGHT: ({waveform_x}, {pos['y']})")
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track robot messages and log expected positions"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🗣️ [Robot-L] message detected")
            self._log_current_position()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🗣️ [Robot-R] message detected")
            self._log_current_position()
    
    def _start_robot_conversation(self):
        """Start simple robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter = "[Robot-L] Hello from LEFT!"
                self.current_speaking_robot = "left"
            else:
                starter = "[Robot-R] Greetings from RIGHT!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🎬 Starting conversation: {starter}")
            asyncio.create_task(self.realtime_client.send_text_message(starter, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = ["[Robot-L] Robot-R, quick question?", "[Robot-L] Hey RIGHT side!"]
                self.current_speaking_robot = "left"
            else:
                topics = ["[Robot-R] Robot-L, interesting.", "[Robot-R] LEFT side, yes?"]
                self.current_speaking_robot = "right"
            
            topic = random.choice(topics)
            logger.info(f"🎭 Next: {topic}")
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            next_conversation = random.uniform(10.0, 15.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple Position Test")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🧪 Starting Simple Position Test...")
    logger.info("🎯 Goal: Verify position logic is working correctly")
    logger.info("📋 This will show in logs where waveform SHOULD be positioned")

    streamer = SimplePositionTestStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 