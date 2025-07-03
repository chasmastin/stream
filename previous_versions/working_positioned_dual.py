#!/usr/bin/env python3
"""
Working Positioned Dual Robot System
- Use direct property access without has_property check
- Actually working GStreamer compositor pad control
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


class WorkingPositionedDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Working version with correct GStreamer compositor pad control
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        logger.info("🤖🌊✅🤖 WorkingPositionedDualRobotStreamer initialized")
        logger.info("✅ Using DIRECT GStreamer property access")
        logger.info(f"📍 Left robot: {self.robot_left_pos}")
        logger.info(f"📍 Right robot: {self.robot_right_pos}")
    
    def start_event_loop(self):
        """Keep Step 3a event loop"""
        logger.info("✅ Setting up Working Positioned Dual Robot API...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 25 words.
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

        # Schedule pipeline discovery and testing
        self.event_loop.call_later(5.0, self._discover_pipeline)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _discover_pipeline(self):
        """Discover pipeline elements and find waveform"""
        GLib.idle_add(self._do_pipeline_discovery)
    
    def _do_pipeline_discovery(self):
        """Find waveform pad using direct property access"""
        try:
            logger.info("🔍 FINDING WAVEFORM PAD WITH DIRECT ACCESS...")
            
            if hasattr(self, 'pipeline') and self.pipeline:
                # Find compositor
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Compositor found!")
                    
                    # List all compositor pads
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        logger.info(f"   🎯 Checking pad: {pad_name}")
                        
                        # Try to access properties directly - compositor pads should have these
                        try:
                            # Just try to get properties - if it works, it's a compositor pad with positioning
                            xpos = pad.get_property("xpos")
                            ypos = pad.get_property("ypos")
                            zorder = pad.get_property("zorder")
                            alpha = pad.get_property("alpha")
                            
                            logger.info(f"      ✅ SUCCESS! Position: ({xpos}, {ypos}), Z: {zorder}, Alpha: {alpha}")
                            
                            # If this is sink_1, it's likely our waveform
                            if "sink_1" in pad_name:
                                self.waveform_pad = pad
                                logger.info(f"   🌊 WAVEFORM PAD FOUND: {pad_name}")
                                logger.info(f"   🌊 Initial position: ({xpos}, {ypos})")
                                
                        except Exception as e:
                            # This pad doesn't have positioning properties (probably src pad)
                            logger.info(f"      ⚠️ No positioning properties (expected for {pad_name}): {str(e)[:50]}")
                    
                    if self.waveform_pad:
                        logger.info("🎉 WAVEFORM PAD SUCCESSFULLY IDENTIFIED!")
                        # Schedule positioning tests
                        GLib.timeout_add(3000, self._test_waveform_positioning)
                    else:
                        logger.error("❌ Could not identify waveform pad!")
                        
                else:
                    logger.error("❌ Compositor not found!")
            else:
                logger.error("❌ Pipeline not available!")
                
        except Exception as e:
            logger.error(f"❌ Pipeline discovery failed: {e}")
        return False
    
    def _test_waveform_positioning(self):
        """Test waveform positioning - dramatic movements first"""
        logger.info("🧪 TESTING WAVEFORM POSITIONING...")
        
        if self.waveform_pad:
            try:
                # Get current position
                current_x = self.waveform_pad.get_property("xpos")
                current_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Current position: ({current_x}, {current_y})")
                
                # Test 1: Move to TOP-LEFT corner (dramatic movement)
                test_x = 100
                test_y = 100
                
                logger.info(f"🌊 TEST 1: Moving to TOP-LEFT corner ({test_x}, {test_y})")
                self.waveform_pad.set_property("xpos", test_x)
                self.waveform_pad.set_property("ypos", test_y)
                
                # Verify the change
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ After move: ({new_x}, {new_y})")
                
                if new_x == test_x and new_y == test_y:
                    logger.info("🎉 TOP-LEFT test SUCCESSFUL!")
                    # Schedule bottom-right test in 4 seconds
                    GLib.timeout_add(4000, self._test_bottom_right)
                else:
                    logger.error(f"❌ TOP-LEFT test FAILED! Expected ({test_x}, {test_y}), got ({new_x}, {new_y})")
                    
            except Exception as e:
                logger.error(f"❌ Positioning test failed: {e}")
        else:
            logger.error("❌ No waveform pad for testing!")
        
        return False
    
    def _test_bottom_right(self):
        """Test bottom-right corner"""
        logger.info("🧪 TEST 2: Moving to BOTTOM-RIGHT corner...")
        
        if self.waveform_pad:
            try:
                test_x = 1000
                test_y = 500
                
                logger.info(f"🌊 Moving to BOTTOM-RIGHT corner ({test_x}, {test_y})")
                self.waveform_pad.set_property("xpos", test_x)
                self.waveform_pad.set_property("ypos", test_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ After move: ({new_x}, {new_y})")
                
                if new_x == test_x and new_y == test_y:
                    logger.info("🎉 BOTTOM-RIGHT test SUCCESSFUL!")
                    logger.info("🎊 DRAMATIC MOVEMENT TESTS PASSED!")
                    # Now test robot positions
                    GLib.timeout_add(4000, self._test_robot_positions)
                else:
                    logger.error(f"❌ BOTTOM-RIGHT test FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Bottom-right test failed: {e}")
        
        return False
    
    def _test_robot_positions(self):
        """Test actual robot positions"""
        logger.info("🤖 TEST 3: Testing ROBOT positions...")
        
        if self.waveform_pad:
            try:
                # Test left robot position
                left_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                left_y = self.robot_left_pos['y']
                
                logger.info(f"🌊 Moving to LEFT robot position ({left_x}, {left_y})")
                self.waveform_pad.set_property("xpos", left_x)
                self.waveform_pad.set_property("ypos", left_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ LEFT robot position: ({new_x}, {new_y})")
                
                # Test right robot position in 4 seconds
                GLib.timeout_add(4000, self._test_right_robot_position)
                
            except Exception as e:
                logger.error(f"❌ Robot position test failed: {e}")
        
        return False
    
    def _test_right_robot_position(self):
        """Test right robot position"""
        logger.info("🤖 TEST 4: Testing RIGHT robot position...")
        
        if self.waveform_pad:
            try:
                right_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                right_y = self.robot_right_pos['y']
                
                logger.info(f"🌊 Moving to RIGHT robot position ({right_x}, {right_y})")
                self.waveform_pad.set_property("xpos", right_x)
                self.waveform_pad.set_property("ypos", right_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ RIGHT robot position: ({new_x}, {new_y})")
                
                logger.info("🎊 ALL POSITIONING TESTS COMPLETE!")
                logger.info("📢 If you see the waveform moving around, positioning is working!")
                logger.info("🎭 Now starting regular robot conversation with positioning...")
                
            except Exception as e:
                logger.error(f"❌ Right robot position test failed: {e}")
        
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track robot and update position"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot will respond - moving waveform LEFT")
            self._update_waveform_for_robot()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot will respond - moving waveform RIGHT")
            self._update_waveform_for_robot()
    
    def _update_waveform_for_robot(self):
        """Update waveform position based on current speaking robot"""
        GLib.idle_add(self._do_robot_waveform_update)
    
    def _do_robot_waveform_update(self):
        """Actually update the waveform position for robot"""
        try:
            if self.waveform_pad:
                if self.current_speaking_robot == "left":
                    # Position for left robot
                    new_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_left_pos['y']
                    logger.info(f"🌊⬅️ Moving waveform to LEFT robot: ({new_x}, {new_y})")
                else:
                    # Position for right robot  
                    new_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_right_pos['y']
                    logger.info(f"🌊➡️ Moving waveform to RIGHT robot: ({new_x}, {new_y})")
                
                # Update position
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify position
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform positioned at: ({actual_x}, {actual_y})")
                
            else:
                logger.warning("⚠️ No waveform pad reference for robot positioning")
                
        except Exception as e:
            logger.error(f"❌ Robot waveform positioning failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! I'm the left robot speaking!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the right robot speaking!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🤖 {self.current_speaking_robot.upper()} robot starting conversation")
            
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, what's your perspective from the right?",
                    "[Robot-L] Hey Robot-R! I'm curious about your thoughts!",
                    "[Robot-L] Robot-R, our positioned dialogue is amazing!",
                ]
                self.current_speaking_robot = "left"
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, from my right position, this is fascinating.",
                    "[Robot-R] Robot-L, your left-side perspective is valuable.",
                    "[Robot-R] Robot-L, our conversation creates great dynamics.",
                ]
                self.current_speaking_robot = "right"
            
            topic = random.choice(conversation_topics)
            logger.info(f"🎭 {self.current_speaking_robot.upper()} robot: {topic[:40]}...")
            
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            next_conversation = random.uniform(12.0, 18.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)

    # Override to disable chat simulation
    def monitor_twitch_chat(self):
        """Disable Twitch chat simulation completely"""
        logger.info("🔇 Twitch chat simulation DISABLED - robots only!")
        # Do nothing - no chat simulation at all
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Working Positioned Dual Robot AI Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖🌊✅🤖 Starting Working Positioned Dual Robot AI Streamer...")
    logger.info("✅ This version uses DIRECT GStreamer property access")
    logger.info("🎯 Should properly detect and control waveform positioning")
    logger.info("🧪 Will test dramatic movements first, then robot positions")

    streamer = WorkingPositionedDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 