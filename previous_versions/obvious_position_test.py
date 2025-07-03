#!/usr/bin/env python3
"""
Obvious Position Test - Use DRAMATICALLY different positions to make movement obvious
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
    """
    Use DRAMATICALLY different positions to make waveform movement obvious
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"] 
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        
        # DRAMATIC position differences - far apart!
        self.LEFT_POSITION = {"x": 50, "y": 50}      # Top-left corner
        self.RIGHT_POSITION = {"x": 1100, "y": 600}  # Bottom-right corner
        
        logger.info("🎭 ObviousPositionStreamer initialized")
        logger.info("🎯 DRAMATIC POSITIONS FOR OBVIOUS MOVEMENT:")
        logger.info(f"   LEFT: {self.LEFT_POSITION} (top-left corner)")
        logger.info(f"   RIGHT: {self.RIGHT_POSITION} (bottom-right corner)")
        logger.info(f"   Video size: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}")
        logger.info("📢 Waveform should JUMP dramatically between corners!")
    
    def start_event_loop(self):
        """Keep Step 3a working setup"""
        logger.info("🎭 Setting up Obvious Position Test...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (LEFT CORNER): Curious, optimistic, in TOP-LEFT corner
        🤖 Robot-R (RIGHT CORNER): Analytical, wise, in BOTTOM-RIGHT corner
        
        Always start messages with [Robot-L] or [Robot-R]. Keep responses under 15 words.
        Reference your corner positions!
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

        # Schedule pipeline debug
        self.event_loop.call_later(5.0, self._debug_and_find_waveform)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _debug_and_find_waveform(self):
        """Find and store waveform pad reference"""
        GLib.idle_add(self._find_waveform_pad)
    
    def _find_waveform_pad(self):
        """Find the waveform pad in the compositor"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Compositor found, looking for waveform pad...")
                    
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        logger.info(f"🔍 Found pad: {pad_name}")
                        
                        # Usually the waveform is sink_1 (background video is sink_0)
                        if "sink_1" in pad_name:
                            self.waveform_pad = pad
                            logger.info(f"🌊 WAVEFORM PAD FOUND: {pad_name}")
                            
                            # Get current position
                            current_x = pad.get_property("xpos")
                            current_y = pad.get_property("ypos")
                            logger.info(f"🌊 Current waveform position: ({current_x}, {current_y})")
                            
                            # Schedule dramatic position test in 3 seconds
                            GLib.timeout_add(3000, self._test_dramatic_movement)
                            break
                else:
                    logger.error("❌ Compositor not found!")
            else:
                logger.error("❌ Pipeline not available!")
        except Exception as e:
            logger.error(f"❌ Error finding waveform pad: {e}")
        return False
    
    def _test_dramatic_movement(self):
        """Test dramatic waveform movement"""
        logger.info("🎪 TESTING DRAMATIC WAVEFORM MOVEMENT!")
        
        if self.waveform_pad:
            try:
                # Move to TOP-LEFT corner
                logger.info(f"🌊 Moving waveform to TOP-LEFT: {self.LEFT_POSITION}")
                self.waveform_pad.set_property("xpos", self.LEFT_POSITION["x"])
                self.waveform_pad.set_property("ypos", self.LEFT_POSITION["y"])
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform moved to: ({new_x}, {new_y})")
                
                # Schedule move to BOTTOM-RIGHT in 5 seconds
                GLib.timeout_add(5000, self._test_move_to_bottom_right)
                
            except Exception as e:
                logger.error(f"❌ Failed to move waveform: {e}")
        else:
            logger.error("❌ No waveform pad available!")
        
        return False
    
    def _test_move_to_bottom_right(self):
        """Move waveform to bottom-right corner"""
        logger.info("🎪 Moving to BOTTOM-RIGHT corner!")
        
        if self.waveform_pad:
            try:
                logger.info(f"🌊 Moving waveform to BOTTOM-RIGHT: {self.RIGHT_POSITION}")
                self.waveform_pad.set_property("xpos", self.RIGHT_POSITION["x"])
                self.waveform_pad.set_property("ypos", self.RIGHT_POSITION["y"])
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform moved to: ({new_x}, {new_y})")
                
                # Schedule alternating movement every 5 seconds
                GLib.timeout_add(5000, self._alternate_position)
                
            except Exception as e:
                logger.error(f"❌ Failed to move to bottom-right: {e}")
        
        return False
    
    def _alternate_position(self):
        """Alternate between corners every 5 seconds"""
        if self.waveform_pad:
            try:
                current_x = self.waveform_pad.get_property("xpos")
                
                if current_x == self.LEFT_POSITION["x"]:
                    # Move to right
                    new_pos = self.RIGHT_POSITION
                    logger.info("🌊 Switching to BOTTOM-RIGHT corner")
                else:
                    # Move to left
                    new_pos = self.LEFT_POSITION
                    logger.info("🌊 Switching to TOP-LEFT corner")
                
                self.waveform_pad.set_property("xpos", new_pos["x"])
                self.waveform_pad.set_property("ypos", new_pos["y"])
                
                # Verify
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform now at: ({actual_x}, {actual_y})")
                
                # Continue alternating
                GLib.timeout_add(5000, self._alternate_position)
                
            except Exception as e:
                logger.error(f"❌ Alternating failed: {e}")
        
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track robot and move to dramatic positions"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🗣️ [Robot-L] -> Moving to TOP-LEFT CORNER")
            self._move_to_left_corner()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🗣️ [Robot-R] -> Moving to BOTTOM-RIGHT CORNER")
            self._move_to_right_corner()
    
    def _move_to_left_corner(self):
        """Move to left corner"""
        GLib.idle_add(self._do_move_left)
    
    def _move_to_right_corner(self):
        """Move to right corner"""
        GLib.idle_add(self._do_move_right)
    
    def _do_move_left(self):
        """Actually move to left corner"""
        if self.waveform_pad:
            try:
                logger.info(f"🌊⬅️ MOVING TO TOP-LEFT: {self.LEFT_POSITION}")
                self.waveform_pad.set_property("xpos", self.LEFT_POSITION["x"])
                self.waveform_pad.set_property("ypos", self.LEFT_POSITION["y"])
                
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ LEFT corner: ({actual_x}, {actual_y})")
            except Exception as e:
                logger.error(f"❌ Move to left failed: {e}")
        return False
    
    def _do_move_right(self):
        """Actually move to right corner"""
        if self.waveform_pad:
            try:
                logger.info(f"🌊➡️ MOVING TO BOTTOM-RIGHT: {self.RIGHT_POSITION}")
                self.waveform_pad.set_property("xpos", self.RIGHT_POSITION["x"])
                self.waveform_pad.set_property("ypos", self.RIGHT_POSITION["y"])
                
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ RIGHT corner: ({actual_x}, {actual_y})")
            except Exception as e:
                logger.error(f"❌ Move to right failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter = "[Robot-L] Hello from TOP-LEFT corner!"
                self.current_speaking_robot = "left"
            else:
                starter = "[Robot-R] Greetings from BOTTOM-RIGHT corner!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🎬 Starting: {starter}")
            asyncio.create_task(self.realtime_client.send_text_message(starter, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = [
                    "[Robot-L] Robot-R, from TOP-LEFT corner here!",
                    "[Robot-L] Hey BOTTOM-RIGHT buddy!"
                ]
                self.current_speaking_robot = "left"
            else:
                topics = [
                    "[Robot-R] Robot-L, from BOTTOM-RIGHT corner!",
                    "[Robot-R] TOP-LEFT friend, hello!"
                ]
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

    parser = argparse.ArgumentParser(description="Obvious Position Test")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🎪 Starting Obvious Position Test...")
    logger.info("🎯 Goal: DRAMATIC waveform movement between corners")
    logger.info("📢 Waveform should JUMP between TOP-LEFT and BOTTOM-RIGHT!")

    streamer = ObviousPositionStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 