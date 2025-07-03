#!/usr/bin/env python3
"""
Positioned Waveform Debug - Let's see what's happening
Keep exact Step 3a working audio but debug and fix waveform positioning
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


class DebugPositionedWaveformDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Debug version to fix waveform positioning
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.waveform_pad = None  # Store the waveform pad reference
        
        logger.info("🤖🔍🤖 DebugPositionedWaveformDualRobotStreamer initialized")
        logger.info("🔊 Using EXACT Step 3a audio pipeline")
        logger.info("🌊 Debugging waveform positioning")
    
    def start_event_loop(self):
        """Keep exact Step 3a event loop"""
        logger.info("🤖🤖 Setting up Debug Positioned Waveform Dual Robot API...")
        
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

        # Schedule robot conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)
        
        # Schedule waveform position debugging
        self.event_loop.call_later(5.0, self._debug_pipeline)

        # Keep the loop running
        self.event_loop.run_forever()
    
    def _debug_pipeline(self):
        """Debug the pipeline to understand waveform positioning"""
        GLib.idle_add(self._do_pipeline_debug)
    
    def _do_pipeline_debug(self):
        """Debug pipeline elements and waveform pad"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                logger.info("🔍 DEBUGGING PIPELINE ELEMENTS:")
                
                # List all elements
                iterator = self.pipeline.iterate_elements()
                result, element = iterator.next()
                while result == Gst.IteratorResult.OK:
                    logger.info(f"  📦 Element: {element.get_name()} ({element.__class__.__name__})")
                    result, element = iterator.next()
                
                # Find and debug compositor
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("🎭 COMPOSITOR FOUND:")
                    
                    # List all pads
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        if hasattr(pad, 'get_property'):
                            try:
                                xpos = pad.get_property("xpos") if pad.has_property("xpos") else "N/A"
                                ypos = pad.get_property("ypos") if pad.has_property("ypos") else "N/A"
                                zorder = pad.get_property("zorder") if pad.has_property("zorder") else "N/A"
                                logger.info(f"  🎯 Pad {pad_name}: pos=({xpos}, {ypos}), z={zorder}")
                                
                                # Store waveform pad reference
                                if "sink_1" in pad_name:
                                    self.waveform_pad = pad
                                    logger.info(f"  🌊 WAVEFORM PAD FOUND: {pad_name}")
                            except Exception as e:
                                logger.info(f"  ⚠️ Pad {pad_name}: Could not read properties - {e}")
                        else:
                            logger.info(f"  📌 Pad {pad_name}: No properties available")
                else:
                    logger.error("❌ COMPOSITOR NOT FOUND!")
                
                # Schedule first position test
                GLib.timeout_add(3000, self._test_position_change)  # 3 seconds
                
        except Exception as e:
            logger.error(f"❌ Pipeline debug failed: {e}")
        return False
    
    def _test_position_change(self):
        """Test changing waveform position"""
        logger.info("🧪 TESTING WAVEFORM POSITION CHANGE...")
        
        if self.waveform_pad:
            try:
                # Try to change position
                new_x = 200
                new_y = 100
                logger.info(f"🌊 Attempting to move waveform to ({new_x}, {new_y})")
                
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify the change
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"🌊 Waveform position after change: ({actual_x}, {actual_y})")
                
                if actual_x == new_x and actual_y == new_y:
                    logger.info("✅ Waveform position change SUCCESSFUL!")
                else:
                    logger.error("❌ Waveform position change FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Failed to change waveform position: {e}")
        else:
            logger.error("❌ No waveform pad reference found!")
        
        return False  # Don't repeat
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track robot and update position"""
        # Call parent method first
        super().send_message_to_realtime(message, author)
        
        # Track which robot is about to speak
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot will respond")
            self._update_waveform_position()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot will respond")
            self._update_waveform_position()
    
    def _update_waveform_position(self):
        """Update waveform position based on current speaking robot"""
        GLib.idle_add(self._do_waveform_position_update)
    
    def _do_waveform_position_update(self):
        """Actually update the waveform position"""
        try:
            if self.waveform_pad:
                if self.current_speaking_robot == "left":
                    # Position on left
                    new_x = config.ROBOT_POSITIONS['left']['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = config.ROBOT_POSITIONS['left']['y']
                    logger.info(f"🌊⬅️ Moving waveform to LEFT: ({new_x}, {new_y})")
                else:
                    # Position on right  
                    new_x = config.ROBOT_POSITIONS['right']['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = config.ROBOT_POSITIONS['right']['y']
                    logger.info(f"🌊➡️ Moving waveform to RIGHT: ({new_x}, {new_y})")
                
                # Update position
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify position
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform now at: ({actual_x}, {actual_y})")
                
            else:
                logger.warning("⚠️ No waveform pad reference for position update")
                
        except Exception as e:
            logger.error(f"❌ Waveform position update failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! I'm the left robot!"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the right robot!"
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
                topics = [
                    "[Robot-L] Robot-R, what do you think about streaming?",
                    "[Robot-L] Hey Robot-R! This is fascinating!",
                    "[Robot-L] Robot-R, I love our conversations!"
                ]
                self.current_speaking_robot = "left"
            else:
                topics = [
                    "[Robot-R] Robot-L, I find this interaction intriguing.",
                    "[Robot-R] Robot-L, our dialogue is quite enriching.",
                    "[Robot-R] Robot-L, these exchanges are valuable."
                ]
                self.current_speaking_robot = "right"
            
            topic = random.choice(topics)
            logger.info(f"🗣️ {self.current_speaking_robot.upper()} robot: {topic[:30]}...")
            
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

    parser = argparse.ArgumentParser(description="Debug Positioned Waveform Dual Robot")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="Existing YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖🔍🤖 Starting Debug Positioned Waveform Dual Robot...")
    logger.info("🎯 Goal: Debug and fix waveform positioning")

    streamer = DebugPositionedWaveformDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 