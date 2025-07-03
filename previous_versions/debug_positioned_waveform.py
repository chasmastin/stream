#!/usr/bin/env python3
"""
Debug Positioned Waveform - Let's see what's happening
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


class DebugPositionedWaveformStreamer(WorkingLiveAIStreamer):
    """Debug version to understand waveform positioning"""
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        
        logger.info("🔍 DebugPositionedWaveformStreamer initialized")
    
    def start_event_loop(self):
        """Step 3a event loop with debugging"""
        logger.info("🔍 Setting up debug dual robot API...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic 
        🤖 Robot-R (Right): Analytical, wise
        
        Always start messages with [Robot-L] or [Robot-R]. Keep responses under 20 words.
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

        # Schedule debugging and conversations
        self.event_loop.call_later(5.0, self._debug_pipeline)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _debug_pipeline(self):
        """Debug the pipeline"""
        GLib.idle_add(self._do_pipeline_debug)
    
    def _do_pipeline_debug(self):
        """Debug pipeline and find waveform"""
        try:
            logger.info("🔍 DEBUGGING PIPELINE:")
            
            if hasattr(self, 'pipeline') and self.pipeline:
                # Find compositor
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Compositor found!")
                    
                    # Check all compositor pads
                    pad_count = 0
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        logger.info(f"🎯 Compositor pad: {pad_name}")
                        
                        try:
                            if hasattr(pad, 'get_property'):
                                xpos = pad.get_property("xpos") if pad.has_property("xpos") else "N/A"
                                ypos = pad.get_property("ypos") if pad.has_property("ypos") else "N/A"
                                logger.info(f"  Position: ({xpos}, {ypos})")
                                
                                # Store waveform pad (usually sink_1)
                                if "sink_1" in pad_name:
                                    self.waveform_pad = pad
                                    logger.info("🌊 WAVEFORM PAD STORED!")
                        except Exception as e:
                            logger.info(f"  Error reading pad properties: {e}")
                        
                        pad_count += 1
                    
                    logger.info(f"📊 Total compositor pads: {pad_count}")
                    
                    # Test waveform positioning in 3 seconds
                    GLib.timeout_add(3000, self._test_waveform_move)
                else:
                    logger.error("❌ Compositor not found!")
            else:
                logger.error("❌ Pipeline not available!")
                
        except Exception as e:
            logger.error(f"❌ Debug failed: {e}")
        return False
    
    def _test_waveform_move(self):
        """Test moving the waveform"""
        logger.info("🧪 TESTING WAVEFORM MOVEMENT...")
        
        if self.waveform_pad:
            try:
                # Get current position
                current_x = self.waveform_pad.get_property("xpos")
                current_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Current waveform position: ({current_x}, {current_y})")
                
                # Move to left position
                left_x = config.ROBOT_POSITIONS['left']['x'] - config.WAVEFORM_WIDTH // 2
                left_y = config.ROBOT_POSITIONS['left']['y']
                
                logger.info(f"🌊 Moving waveform to LEFT: ({left_x}, {left_y})")
                self.waveform_pad.set_property("xpos", left_x)
                self.waveform_pad.set_property("ypos", left_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Waveform position after move: ({new_x}, {new_y})")
                
                if new_x == left_x and new_y == left_y:
                    logger.info("✅ WAVEFORM MOVEMENT SUCCESSFUL!")
                    # Schedule move to right in 5 seconds
                    GLib.timeout_add(5000, self._test_move_right)
                else:
                    logger.error("❌ WAVEFORM MOVEMENT FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Waveform move test failed: {e}")
        else:
            logger.error("❌ No waveform pad available for testing!")
        
        return False
    
    def _test_move_right(self):
        """Test moving waveform to right"""
        logger.info("🧪 TESTING MOVE TO RIGHT...")
        
        if self.waveform_pad:
            try:
                right_x = config.ROBOT_POSITIONS['right']['x'] - config.WAVEFORM_WIDTH // 2
                right_y = config.ROBOT_POSITIONS['right']['y']
                
                logger.info(f"🌊 Moving waveform to RIGHT: ({right_x}, {right_y})")
                self.waveform_pad.set_property("xpos", right_x)
                self.waveform_pad.set_property("ypos", right_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Waveform position after RIGHT move: ({new_x}, {new_y})")
                
                if new_x == right_x and new_y == right_y:
                    logger.info("✅ RIGHT MOVEMENT SUCCESSFUL!")
                else:
                    logger.error("❌ RIGHT MOVEMENT FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Right move test failed: {e}")
        
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track robot and update position"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot will respond")
            self._move_waveform_to_robot()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot will respond")
            self._move_waveform_to_robot()
    
    def _move_waveform_to_robot(self):
        """Move waveform to current robot position"""
        GLib.idle_add(self._do_move_waveform)
    
    def _do_move_waveform(self):
        """Actually move the waveform"""
        if self.waveform_pad:
            try:
                if self.current_speaking_robot == "left":
                    new_x = config.ROBOT_POSITIONS['left']['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = config.ROBOT_POSITIONS['left']['y']
                    logger.info(f"🌊⬅️ Moving to LEFT: ({new_x}, {new_y})")
                else:
                    new_x = config.ROBOT_POSITIONS['right']['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = config.ROBOT_POSITIONS['right']['y']
                    logger.info(f"🌊➡️ Moving to RIGHT: ({new_x}, {new_y})")
                
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform moved to: ({actual_x}, {actual_y})")
                
            except Exception as e:
                logger.error(f"❌ Waveform move failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! Left robot here!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! Right robot speaking!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🤖 {self.current_speaking_robot.upper()} robot starting")
            
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = ["[Robot-L] Robot-R, what do you think?", "[Robot-L] Hey Robot-R!"]
                self.current_speaking_robot = "left"
            else:
                topics = ["[Robot-R] Robot-L, interesting point.", "[Robot-R] Indeed, Robot-L."]
                self.current_speaking_robot = "right"
            
            topic = random.choice(topics)
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            self._switch_robot_voice()
            
            next_conversation = random.uniform(12.0, 18.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)
    
    def _switch_robot_voice(self):
        """Switch robot voice"""
        current_index = self.robot_voices.index(self.current_robot_voice)
        next_index = (current_index + 1) % len(self.robot_voices)
        self.current_robot_voice = self.robot_voices[next_index]
        logger.info(f"🔄 Next voice: {self.current_robot_voice}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug Positioned Waveform")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🔍 Starting Debug Positioned Waveform...")
    logger.info("🎯 Goal: Debug waveform positioning issues")

    streamer = DebugPositionedWaveformStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 