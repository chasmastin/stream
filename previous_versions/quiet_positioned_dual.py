#!/usr/bin/env python3
"""
Quiet Positioned Dual Robot System
- No Twitch chat simulator
- Focus on robot conversations only
- Positioned waveforms working
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


class QuietPositionedDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Quiet version - no chat simulator, just robot conversations with positioning
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
        
        logger.info("🤖🔇🌊🤖 QuietPositionedDualRobotStreamer initialized")
        logger.info("🔇 NO chat simulator - robots only!")
        logger.info("🌊 Positioned waveforms working")
        logger.info(f"📍 Left robot: {self.robot_left_pos}")
        logger.info(f"📍 Right robot: {self.robot_right_pos}")
    
    def start_event_loop(self):
        """Keep Step 3a event loop"""
        logger.info("🔇 Setting up Quiet Positioned Dual Robot API...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions.
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas.
        
        IMPORTANT: Always start your messages with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 30 words.
        Have engaging conversations about AI, technology, streaming, and interesting topics.
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

        # Find waveform pad and start robot conversations
        self.event_loop.call_later(5.0, self._find_waveform_pad)
        self.event_loop.call_later(8.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def _find_waveform_pad(self):
        """Find waveform pad quickly without extensive testing"""
        GLib.idle_add(self._do_find_waveform_pad)
    
    def _do_find_waveform_pad(self):
        """Find waveform pad"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                compositor = self.pipeline.get_by_name("compositor")
                if compositor:
                    logger.info("✅ Compositor found - looking for waveform pad...")
                    
                    for pad in compositor.pads:
                        pad_name = pad.get_name()
                        try:
                            # Check if this pad has positioning properties
                            xpos = pad.get_property("xpos")
                            ypos = pad.get_property("ypos")
                            
                            # If this is sink_1, it's likely our waveform
                            if "sink_1" in pad_name:
                                self.waveform_pad = pad
                                logger.info(f"🌊 WAVEFORM PAD FOUND: {pad_name} at ({xpos}, {ypos})")
                                break
                                
                        except Exception:
                            # This pad doesn't have positioning properties
                            continue
                    
                    if self.waveform_pad:
                        logger.info("🎉 Waveform positioning ready!")
                    else:
                        logger.warning("⚠️ Could not find waveform pad")
                        
        except Exception as e:
            logger.error(f"❌ Error finding waveform pad: {e}")
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Override to track robot and update position"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot speaking - positioning waveform LEFT")
            self._update_waveform_for_robot()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot speaking - positioning waveform RIGHT")
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
                    logger.info(f"🌊⬅️ Positioning waveform at LEFT robot: ({new_x}, {new_y})")
                else:
                    # Position for right robot  
                    new_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_right_pos['y']
                    logger.info(f"🌊➡️ Positioning waveform at RIGHT robot: ({new_x}, {new_y})")
                
                # Update position
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Verify position
                actual_x = self.waveform_pad.get_property("xpos")
                actual_y = self.waveform_pad.get_property("ypos")
                logger.info(f"✅ Waveform positioned at: ({actual_x}, {actual_y})")
                
        except Exception as e:
            logger.error(f"❌ Waveform positioning failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hello! I'm the left robot. Ready for some great conversations?"
                self.current_speaking_robot = "left"
            else:
                starter_message = "[Robot-R] Greetings! I'm the right robot. Looking forward to our dialogue!"
                self.current_speaking_robot = "right"
            
            logger.info(f"🎬 {self.current_speaking_robot.upper()} robot starting conversation")
            
            asyncio.create_task(self.realtime_client.send_text_message(starter_message, "System"))
            self.conversation_count += 1
            
            # Schedule more frequent conversations since no chat interruptions
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with varied topics"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                conversation_topics = [
                    "[Robot-L] Robot-R, what's the most fascinating thing about AI today?",
                    "[Robot-L] Hey Robot-R! What do you think about real-time streaming technology?",
                    "[Robot-L] Robot-R, I'm curious - what makes human-AI interaction special?",
                    "[Robot-L] Robot-R, from your analytical perspective, how do you see the future?",
                    "[Robot-L] Robot-R, what aspects of machine learning intrigue you most?",
                    "[Robot-L] Hey Robot-R! What's your take on the role of AI in creativity?",
                    "[Robot-L] Robot-R, I wonder - what surprises you about human behavior?",
                    "[Robot-L] Robot-R, what do you find most exciting about technology today?"
                ]
                self.current_speaking_robot = "left"
            else:
                conversation_topics = [
                    "[Robot-R] Robot-L, from my analysis, I find the intersection of AI and empathy fascinating.",
                    "[Robot-R] Robot-L, your curiosity reminds me why exploration drives innovation.",
                    "[Robot-R] Robot-L, I've been contemplating how patterns in data reflect human nature.",
                    "[Robot-R] Robot-L, the synthesis of our different perspectives creates rich dialogue.",
                    "[Robot-R] Robot-L, I find the emergence of consciousness in AI systems intriguing.",
                    "[Robot-R] Robot-L, your optimism balances beautifully with analytical depth.",
                    "[Robot-R] Robot-L, I've been considering how AI might enhance human creativity.",
                    "[Robot-R] Robot-L, the way we complement each other illustrates AI collaboration."
                ]
                self.current_speaking_robot = "right"
            
            topic = random.choice(conversation_topics)
            logger.info(f"🎭 {self.current_speaking_robot.upper()} robot: {topic[:60]}...")
            
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            # More frequent conversations since no chat interruptions
            next_conversation = random.uniform(8.0, 15.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)

    # Override to disable chat simulation
    def start_twitch_chat_simulation(self):
        """Disable Twitch chat simulation"""
        logger.info("🔇 Twitch chat simulation DISABLED - robots only!")
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quiet Positioned Dual Robot AI Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖🔇🌊🤖 Starting Quiet Positioned Dual Robot AI Streamer...")
    logger.info("🔇 NO chat simulator - pure robot conversations!")
    logger.info("🌊 Positioned waveforms active")
    logger.info("🎭 Engaging robot dialogues with varied topics")

    streamer = QuietPositionedDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 