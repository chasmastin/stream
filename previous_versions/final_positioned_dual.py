#!/usr/bin/env python3
"""
Final Positioned Dual Robot System
- Use Step 3a proven working base
- Add clear waveform positioning with obvious movements
- Test with both subtle and dramatic position changes
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


class FinalPositionedDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Final positioned dual robot system
    - Keep exact Step 3a working foundation
    - Add clear waveform positioning
    - Test both subtle robot positions and dramatic corner positions
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None
        self.test_mode = "robot_positions"  # or "dramatic_corners"
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        # Dramatic corner positions for testing
        self.corner_left_pos = {"x": 100, "y": 100}
        self.corner_right_pos = {"x": 1000, "y": 500}
        
        logger.info("🤖🌊🌊🤖 FinalPositionedDualRobotStreamer initialized")
        logger.info("🔊 Using EXACT Step 3a working audio pipeline")
        logger.info("🎯 Adding positioned waveforms with testing modes")
        logger.info("")
        logger.info("📍 ROBOT POSITIONS:")
        logger.info(f"   Left robot: {self.robot_left_pos}")
        logger.info(f"   Right robot: {self.robot_right_pos}")
        logger.info("")
        logger.info("📍 CORNER TEST POSITIONS:")
        logger.info(f"   Left corner: {self.corner_left_pos}")
        logger.info(f"   Right corner: {self.corner_right_pos}")
        logger.info("")
        logger.info(f"🎮 Starting in mode: {self.test_mode}")
    
    def start_event_loop(self):
        """Keep exact Step 3a event loop"""
        logger.info("🤖🤖 Setting up Final Positioned Dual Robot API...")
        
        # Use the exact same threading setup as Step 3a
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Enhanced dual robot instructions
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 25 words.
        
        Example conversation:
        [Robot-L] Hey Robot-R! What's your take on AI streaming?
        [Robot-R] Robot-L, I find the real-time interaction fascinating.
        [Robot-L] I love how we complement each other!
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
            self.audio_received_callback,  # Keep parent's callback
            self.audio_level_callback      # Keep parent's callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule pipeline discovery and testing
        self.event_loop.call_later(5.0, self._discover_pipeline)
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def _discover_pipeline(self):
        """Discover pipeline elements and find waveform"""
        GLib.idle_add(self._do_pipeline_discovery)
    
    def _do_pipeline_discovery(self):
        """Find and store waveform pad reference"""
        try:
            logger.info("🔍 DISCOVERING PIPELINE ELEMENTS...")
            
            if hasattr(self, 'pipeline') and self.pipeline:
                # Find compositor
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Compositor found!")
                    
                    # List all compositor pads
                    pad_count = 0
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        logger.info(f"   🎯 Pad: {pad_name}")
                        
                        try:
                            if hasattr(pad, 'get_property'):
                                xpos = pad.get_property("xpos") if pad.has_property("xpos") else "N/A"
                                ypos = pad.get_property("ypos") if pad.has_property("ypos") else "N/A"
                                zorder = pad.get_property("zorder") if pad.has_property("zorder") else "N/A"
                                alpha = pad.get_property("alpha") if pad.has_property("alpha") else "N/A"
                                logger.info(f"      Position: ({xpos}, {ypos}), Z: {zorder}, Alpha: {alpha}")
                                
                                # Store waveform pad (usually sink_1)
                                if "sink_1" in pad_name:
                                    self.waveform_pad = pad
                                    logger.info(f"   🌊 WAVEFORM PAD IDENTIFIED: {pad_name}")
                        except Exception as e:
                            logger.info(f"      Error reading properties: {e}")
                        
                        pad_count += 1
                    
                    logger.info(f"📊 Total compositor pads: {pad_count}")
                    
                    if self.waveform_pad:
                        # Schedule positioning tests
                        GLib.timeout_add(3000, self._test_basic_positioning)
                    else:
                        logger.error("❌ No waveform pad found!")
                        
                else:
                    logger.error("❌ Compositor not found!")
            else:
                logger.error("❌ Pipeline not available!")
                
        except Exception as e:
            logger.error(f"❌ Pipeline discovery failed: {e}")
        return False
    
    def _test_basic_positioning(self):
        """Test basic waveform positioning"""
        logger.info("🧪 TESTING BASIC WAVEFORM POSITIONING...")
        
        if self.waveform_pad:
            try:
                # Get current position
                current_x = self.waveform_pad.get_property("xpos")
                current_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Current waveform position: ({current_x}, {current_y})")
                
                # Test 1: Move to left robot position
                left_waveform_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                left_waveform_y = self.robot_left_pos['y']
                
                logger.info(f"🌊 TEST 1: Moving to LEFT robot position ({left_waveform_x}, {left_waveform_y})")
                self.waveform_pad.set_property("xpos", left_waveform_x)
                self.waveform_pad.set_property("ypos", left_waveform_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform moved to: ({new_x}, {new_y})")
                
                if new_x == left_waveform_x and new_y == left_waveform_y:
                    logger.info("✅ LEFT robot position test SUCCESSFUL!")
                    # Schedule test 2 in 4 seconds
                    GLib.timeout_add(4000, self._test_right_position)
                else:
                    logger.error("❌ LEFT robot position test FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Basic positioning test failed: {e}")
        else:
            logger.error("❌ No waveform pad for testing!")
        
        return False
    
    def _test_right_position(self):
        """Test moving to right robot position"""
        logger.info("🧪 TEST 2: Moving to RIGHT robot position...")
        
        if self.waveform_pad:
            try:
                right_waveform_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                right_waveform_y = self.robot_right_pos['y']
                
                logger.info(f"🌊 Moving to RIGHT robot position ({right_waveform_x}, {right_waveform_y})")
                self.waveform_pad.set_property("xpos", right_waveform_x)
                self.waveform_pad.set_property("ypos", right_waveform_y)
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform moved to: ({new_x}, {new_y})")
                
                if new_x == right_waveform_x and new_y == right_waveform_y:
                    logger.info("✅ RIGHT robot position test SUCCESSFUL!")
                    logger.info("🎉 ROBOT POSITIONING TESTS PASSED!")
                    
                    # Now test dramatic corners in 4 seconds
                    GLib.timeout_add(4000, self._test_dramatic_corners)
                else:
                    logger.error("❌ RIGHT robot position test FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Right position test failed: {e}")
        
        return False
    
    def _test_dramatic_corners(self):
        """Test dramatic corner positioning"""
        logger.info("🎪 TEST 3: Testing DRAMATIC corner positioning...")
        
        if self.waveform_pad:
            try:
                logger.info(f"🌊 Moving to DRAMATIC LEFT corner {self.corner_left_pos}")
                self.waveform_pad.set_property("xpos", self.corner_left_pos["x"])
                self.waveform_pad.set_property("ypos", self.corner_left_pos["y"])
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Dramatic left corner: ({new_x}, {new_y})")
                
                # Schedule dramatic right corner in 4 seconds
                GLib.timeout_add(4000, self._test_dramatic_right_corner)
                
            except Exception as e:
                logger.error(f"❌ Dramatic corner test failed: {e}")
        
        return False
    
    def _test_dramatic_right_corner(self):
        """Test dramatic right corner"""
        logger.info("🎪 TEST 4: Moving to DRAMATIC RIGHT corner...")
        
        if self.waveform_pad:
            try:
                logger.info(f"🌊 Moving to DRAMATIC RIGHT corner {self.corner_right_pos}")
                self.waveform_pad.set_property("xpos", self.corner_right_pos["x"])
                self.waveform_pad.set_property("ypos", self.corner_right_pos["y"])
                
                # Verify
                new_x = self.waveform_pad.get_property("xpos")
                new_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Dramatic right corner: ({new_x}, {new_y})")
                
                logger.info("🎉 ALL POSITIONING TESTS COMPLETE!")
                logger.info("📢 If you can see the waveform jumping around, positioning is working!")
                logger.info("📢 If waveform stays in center, there may be a visual update issue")
                
                # Start alternating between positions every 6 seconds for visual confirmation
                GLib.timeout_add(6000, self._alternate_test_positions)
                
            except Exception as e:
                logger.error(f"❌ Dramatic right corner test failed: {e}")
        
        return False
    
    def _alternate_test_positions(self):
        """Alternate between test positions for visual confirmation"""
        if self.waveform_pad:
            try:
                current_x = self.waveform_pad.get_property("xpos")
                
                if current_x <= 200:  # Currently at left side
                    # Move to right
                    new_pos = self.corner_right_pos
                    logger.info("🔄 Visual test: Moving to RIGHT corner")
                else:
                    # Move to left
                    new_pos = self.corner_left_pos
                    logger.info("🔄 Visual test: Moving to LEFT corner")
                
                self.waveform_pad.set_property("xpos", new_pos["x"])
                self.waveform_pad.set_property("ypos", new_pos["y"])
                
                # Verify
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Visual test position: ({actual_x}, {actual_y})")
                
                # Continue alternating for visual confirmation
                GLib.timeout_add(6000, self._alternate_test_positions)
                
            except Exception as e:
                logger.error(f"❌ Visual test failed: {e}")
        
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track robot and update position"""
        # Call parent method first
        super().send_message_to_realtime(message, author)
        
        # Track which robot is about to speak and update position
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot will respond")
            self._update_waveform_for_robot()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot will respond")
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
                logger.info(f"✅ Robot waveform positioned at: ({actual_x}, {actual_y})")
                
            else:
                logger.warning("⚠️ No waveform pad reference for robot positioning")
                
        except Exception as e:
            logger.error(f"❌ Robot waveform positioning failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! I'm the left robot speaking from my position!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the right robot speaking from my position!"
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
                    "[Robot-L] Robot-R, what's your perspective from the right side?",
                    "[Robot-L] Hey Robot-R! I'm curious about your thoughts!",
                    "[Robot-L] Robot-R, our positioned conversations are fascinating!",
                    "[Robot-L] From the left here, Robot-R, what do you observe?",
                    "[Robot-L] Robot-R, I love our back-and-forth dialogue!"
                ]
                self.current_speaking_robot = "left"
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, from my right position, I find this intriguing.",
                    "[Robot-R] Robot-L, our positioned dialogue creates interesting dynamics.",
                    "[Robot-R] Robot-L, I appreciate your left-side perspective.",
                    "[Robot-R] From the right side, Robot-L, our conversations are enriching.",
                    "[Robot-R] Robot-L, the spatial aspect of our chat is fascinating."
                ]
                self.current_speaking_robot = "right"
            
            # Pick a random conversation topic for the current robot
            topic = random.choice(conversation_topics)
            logger.info(f"🎭 {self.current_speaking_robot.upper()} robot preparing: {topic[:50]}...")
            
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
        description="Final Positioned Dual Robot AI Streamer"
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

    logger.info("🤖🌊🌊🤖 Starting Final Positioned Dual Robot AI Streamer...")
    logger.info("📝 This version combines:")
    logger.info("   ✅ Step 3a working audio foundation")
    logger.info("   🎯 Positioned waveforms for left/right robots")
    logger.info("   🧪 Comprehensive positioning tests")
    logger.info("   📢 Clear logging to verify functionality")
    logger.info("")
    logger.info("🔍 Watch the logs to see if waveform positioning is working!")
    logger.info("🎪 The system will test both robot positions and dramatic corners")

    # Use our final positioned dual robot streamer
    streamer = FinalPositionedDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 