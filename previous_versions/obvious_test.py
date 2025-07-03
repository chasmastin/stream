#!/usr/bin/env python3
"""
Obvious Position Test - DRAMATIC waveform movement
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


class ObviousPositionStreamer(WorkingLiveAIStreamer):
    """DRAMATIC position changes to make movement obvious"""
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"] 
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        
        # EXTREME positions - corners of screen!
        self.LEFT_POSITION = {"x": 50, "y": 50}      # Top-left
        self.RIGHT_POSITION = {"x": 1100, "y": 600}  # Bottom-right
        
        logger.info("🎪 DRAMATIC POSITION TEST!")
        logger.info(f"   LEFT: {self.LEFT_POSITION} (top-left)")
        logger.info(f"   RIGHT: {self.RIGHT_POSITION} (bottom-right)")
        logger.info("🎯 Waveform should JUMP between corners!")
    
    def start_event_loop(self):
        """Step 3a setup with dramatic positioning"""
        logger.info("🎪 Setting up Dramatic Position Test...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are dual robots in EXTREME corners:
        
        🤖 Robot-L: TOP-LEFT corner (50, 50)
        🤖 Robot-R: BOTTOM-RIGHT corner (1100, 600)
        
        Always start with [Robot-L] or [Robot-R]. Keep under 15 words.
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

        # Find waveform and test movement
        self.event_loop.call_later(5.0, self._find_waveform)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _find_waveform(self):
        """Find waveform pad"""
        GLib.idle_add(self._do_find_waveform)
    
    def _do_find_waveform(self):
        """Find the waveform compositor pad"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Found compositor")
                    
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        if "sink_1" in pad_name:  # Usually waveform
                            self.waveform_pad = pad
                            logger.info(f"🌊 WAVEFORM PAD: {pad_name}")
                            
                            # Test dramatic movement in 3 seconds
                            GLib.timeout_add(3000, self._test_extreme_movement)
                            break
                else:
                    logger.error("❌ No compositor!")
        except Exception as e:
            logger.error(f"❌ Find waveform failed: {e}")
        return False
    
    def _test_extreme_movement(self):
        """Test EXTREME waveform movement"""
        logger.info("🎪 TESTING EXTREME MOVEMENT!")
        
        if self.waveform_pad:
            try:
                # Move to TOP-LEFT
                logger.info("🌊 Moving to TOP-LEFT CORNER (50, 50)")
                self.waveform_pad.set_property("xpos", 50)
                self.waveform_pad.set_property("ypos", 50)
                
                # Verify
                x = self.waveform_pad.get_property("xpos")
                y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Moved to: ({x}, {y})")
                
                # Move to BOTTOM-RIGHT in 5 seconds
                GLib.timeout_add(5000, self._move_to_bottom_right)
                
            except Exception as e:
                logger.error(f"❌ Movement failed: {e}")
        else:
            logger.error("❌ No waveform pad!")
        
        return False
    
    def _move_to_bottom_right(self):
        """Move to bottom-right"""
        if self.waveform_pad:
            try:
                logger.info("🌊 Moving to BOTTOM-RIGHT CORNER (1100, 600)")
                self.waveform_pad.set_property("xpos", 1100)
                self.waveform_pad.set_property("ypos", 600)
                
                x = self.waveform_pad.get_property("xpos")
                y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Moved to: ({x}, {y})")
                
                # Continue alternating every 5 seconds
                GLib.timeout_add(5000, self._alternate_corners)
                
            except Exception as e:
                logger.error(f"❌ Move failed: {e}")
        return False
    
    def _alternate_corners(self):
        """Alternate between corners"""
        if self.waveform_pad:
            try:
                current_x = self.waveform_pad.get_property("xpos")
                
                if current_x <= 100:  # Currently at left
                    logger.info("🌊 JUMPING to BOTTOM-RIGHT!")
                    self.waveform_pad.set_property("xpos", 1100)
                    self.waveform_pad.set_property("ypos", 600)
                else:  # Currently at right
                    logger.info("🌊 JUMPING to TOP-LEFT!")
                    self.waveform_pad.set_property("xpos", 50)
                    self.waveform_pad.set_property("ypos", 50)
                
                x = self.waveform_pad.get_property("xpos")
                y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Now at: ({x}, {y})")
                
                # Continue alternating
                GLib.timeout_add(5000, self._alternate_corners)
                
            except Exception as e:
                logger.error(f"❌ Alternate failed: {e}")
        
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Move based on robot messages"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            logger.info("🗣️ [Robot-L] -> TOP-LEFT CORNER!")
            self._jump_to_left()
        elif "[Robot-R]" in message:
            logger.info("🗣️ [Robot-R] -> BOTTOM-RIGHT CORNER!")
            self._jump_to_right()
    
    def _jump_to_left(self):
        GLib.idle_add(self._do_jump_left)
    
    def _jump_to_right(self):
        GLib.idle_add(self._do_jump_right)
    
    def _do_jump_left(self):
        if self.waveform_pad:
            try:
                logger.info("🌊⬅️ JUMPING TO TOP-LEFT!")
                self.waveform_pad.set_property("xpos", 50)
                self.waveform_pad.set_property("ypos", 50)
                
                x = self.waveform_pad.get_property("xpos")
                y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ LEFT: ({x}, {y})")
            except Exception as e:
                logger.error(f"❌ Jump left failed: {e}")
        return False
    
    def _do_jump_right(self):
        if self.waveform_pad:
            try:
                logger.info("🌊➡️ JUMPING TO BOTTOM-RIGHT!")
                self.waveform_pad.set_property("xpos", 1100)
                self.waveform_pad.set_property("ypos", 600)
                
                x = self.waveform_pad.get_property("xpos")
                y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ RIGHT: ({x}, {y})")
            except Exception as e:
                logger.error(f"❌ Jump right failed: {e}")
        return False

    def _start_robot_conversation(self):
        """Start conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter = "[Robot-L] Hello from TOP-LEFT!"
            else:
                starter = "[Robot-R] Greetings from BOTTOM-RIGHT!"
            
            asyncio.create_task(self.realtime_client.send_text_message(starter, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(10.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = ["[Robot-L] From TOP-LEFT corner!", "[Robot-L] Hey BOTTOM-RIGHT!"]
            else:
                topics = ["[Robot-R] From BOTTOM-RIGHT!", "[Robot-R] Hi TOP-LEFT!"]
            
            topic = random.choice(topics)
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            self.event_loop.call_later(8.0, self._continue_robot_conversation)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dramatic Position Test")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🎪 DRAMATIC POSITION TEST!")
    logger.info("🎯 Waveform should JUMP between screen corners!")

    streamer = ObviousPositionStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 