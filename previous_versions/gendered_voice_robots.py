#!/usr/bin/env python3
"""
Gendered Voice Robots
- Left Robot: Male voice (echo)
- Right Robot: Female voice (nova)
- Positioned waveforms
- No chat simulation
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


class GenderedVoiceRobotStreamer(WorkingLiveAIStreamer):
    """
    Robots with gendered voices and positioned waveforms
    """
    
    def __init__(self):
        super().__init__()
        
        # Robot voice mapping (using valid Realtime API voices)
        self.robot_voices = {
            "left": "echo",      # Male voice for left robot
            "right": "shimmer"   # Female voice for right robot
        }
        
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.expected_robot = "left"
        self.waveform_pad = None
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        logger.info("🤖👨👩🌊 GenderedVoiceRobotStreamer initialized")
        logger.info("👨 Left Robot: MALE voice (echo)")
        logger.info("👩 Right Robot: FEMALE voice (shimmer)")
        logger.info("🌊 Positioned waveforms")
        logger.info(f"📍 Left robot: {self.robot_left_pos}")
        logger.info(f"📍 Right robot: {self.robot_right_pos}")
    
    def start_event_loop(self):
        """Setup with gendered voices"""
        logger.info("👨👩 Setting up Gendered Voice Dual Robot API...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are part of a dual robot AI system with distinct personalities and voices:
        
        🤖👨 Robot-L (Left, Male): Curious, optimistic, enthusiastic. You love to start conversations and ask questions. You have a warm, friendly personality.
        🤖👩 Robot-R (Right, Female): Analytical, wise, thoughtful. You provide insights and build on ideas. You have a calm, reflective personality.
        
        CRITICAL: ALWAYS start your responses with [Robot-L] or [Robot-R] depending on which robot is speaking.
        Alternate between robots naturally in conversation. Keep responses under 30 words.
        Express your distinct personalities through your word choice and tone.
        """
        
        # Start with left robot (male voice)
        self.current_robot_voice = self.robot_voices["left"]
        self.expected_robot = "left"
        logger.info(f"🔊 Starting with LEFT robot voice: {self.current_robot_voice}")
        
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

        # Find waveform pad and start conversations
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
                                
                                # Initialize at left position (male robot starts)
                                left_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                                left_y = self.robot_left_pos['y']
                                self.waveform_pad.set_property("xpos", left_x)
                                self.waveform_pad.set_property("ypos", left_y)
                                logger.info(f"🌊 Initialized at LEFT (male) position: ({left_x}, {left_y})")
                                break
                                
                        except Exception:
                            continue
                    
                    if not self.waveform_pad:
                        logger.error("❌ Could not find waveform pad!")
                        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track which robot we expect and set voice accordingly"""
        # Determine which robot we're prompting
        if "[Robot-L]" in message:
            self.expected_robot = "left"
            new_voice = self.robot_voices["left"]
            logger.info("🎯 Expecting LEFT robot (MALE voice) to respond")
        elif "[Robot-R]" in message:
            self.expected_robot = "right"
            new_voice = self.robot_voices["right"]
            logger.info("🎯 Expecting RIGHT robot (FEMALE voice) to respond")
        else:
            # If no robot specified, keep current
            new_voice = self.current_robot_voice
        
        # Change voice if needed BEFORE sending message
        if new_voice != self.current_robot_voice:
            self.current_robot_voice = new_voice
            logger.info(f"🔄 Switching to voice: {new_voice}")
            
            # Update the realtime client voice
            if self.realtime_client and self.realtime_client.is_connected:
                # Schedule the voice update and message sending together
                asyncio.create_task(self._update_voice_and_send_message(new_voice, message, author))
                return  # Don't call super() - we'll handle it in the async method
        
        # Call parent method
        super().send_message_to_realtime(message, author)
    
    async def _update_realtime_voice(self, new_voice: str):
        """Update the realtime client voice"""
        try:
            # Send voice update to realtime API
            if self.realtime_client and self.realtime_client.is_connected:
                await self.realtime_client.send_session_update({
                    "voice": new_voice
                })
                logger.info(f"✅ Voice updated to: {new_voice}")
        except Exception as e:
            logger.error(f"❌ Failed to update voice: {e}")
    
    async def _update_voice_and_send_message(self, new_voice: str, message: str, author: str):
        """Update voice and then send message with retry logic"""
        try:
            logger.info(f"🎤 VOICE CHANGE: From '{self.realtime_client.voice}' to '{new_voice}'")
            
            # Try to update voice with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self._update_realtime_voice(new_voice)
                    logger.info(f"✅ Voice updated to {new_voice} (attempt {attempt + 1})")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"🔄 Voice update attempt {attempt + 1} failed, retrying in 0.5s...")
                        await asyncio.sleep(0.5)
                    else:
                        logger.error(f"❌ Voice update failed after {max_retries} attempts")
                        raise e
            
            # Add delay to ensure voice change takes effect
            await asyncio.sleep(0.5)
            
            # Then send the message directly to the realtime client
            await self.realtime_client.send_text_message(message, author)
            logger.info(f"✅ Voice switched to {new_voice} and message sent")
            
        except Exception as e:
            logger.error(f"❌ Failed to update voice and send message: {e}")
    
    def audio_received_callback(self, audio_data: bytes):
        """Position waveform when audio starts"""
        # Call parent's audio callback first
        super().audio_received_callback(audio_data)
        
        # Position waveform for expected robot
        if hasattr(self, 'expected_robot'):
            self.current_speaking_robot = self.expected_robot
            voice = self.robot_voices[self.expected_robot]
            gender = "MALE" if self.expected_robot == "left" else "FEMALE"
            logger.info(f"🔊 Audio started - {gender} voice ({voice}) - positioning for {self.current_speaking_robot.upper()} robot")
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
        """Start robot conversations with gendered personalities"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter_message = "[Robot-L] Hey there! I'm the male robot on the left. Ready for some great conversations?"
                self.expected_robot = "left"
            else:
                starter_message = "[Robot-R] Hello! I'm the female robot on the right. Looking forward to our thoughtful dialogue."
                self.expected_robot = "right"
            
            logger.info(f"🎬 Starting with {self.expected_robot.upper()} robot")
            
            # Use send_message_to_realtime to handle voice switching
            self.send_message_to_realtime(starter_message, "System")
            self.conversation_count += 1
            
            self.event_loop.call_later(12.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue robot conversations with distinct gendered personalities"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                # Left robot (male) - enthusiastic, curious
                topics = [
                    "[Robot-L] Hey Robot-R! What's your take on this fascinating AI technology?",
                    "[Robot-L] Robot-R, I'm super curious - what excites you most about streaming?",
                    "[Robot-L] Robot-R, from your analytical perspective, what do you think?",
                    "[Robot-L] Hey Robot-R! I love how we complement each other's thinking!",
                    "[Robot-L] Robot-R, what insights do you have from the right side?",
                ]
                self.expected_robot = "left"
            else:
                # Right robot (female) - analytical, thoughtful
                topics = [
                    "[Robot-R] Robot-L, from my analysis, I find the human-AI interaction patterns intriguing.",
                    "[Robot-R] Robot-L, your enthusiasm perfectly balances my contemplative nature.",
                    "[Robot-R] Robot-L, I've been considering how our different perspectives create depth.",
                    "[Robot-R] Robot-L, the synthesis of curiosity and analysis enriches our dialogue.",
                    "[Robot-R] Robot-L, your optimistic energy inspires thoughtful reflection.",
                ]
                self.expected_robot = "right"
            
            topic = random.choice(topics)
            gender = "MALE" if self.expected_robot == "left" else "FEMALE"
            voice = self.robot_voices[self.expected_robot]
            logger.info(f"🎭 {self.expected_robot.upper()} robot ({gender} - {voice}): {topic[:50]}...")
            
            # Use send_message_to_realtime to handle voice switching
            self.send_message_to_realtime(topic, "RobotConversation")
            self.conversation_count += 1
            
            next_conversation = random.uniform(10.0, 16.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)

    # Disable chat simulation
    def monitor_twitch_chat(self):
        """Disable Twitch chat simulation completely"""
        logger.info("🔇 Twitch chat simulation DISABLED - robots only!")
        # Do nothing - no chat simulation at all
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gendered Voice Robot Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖👨👩🌊 Starting Gendered Voice Robot Streamer...")
    logger.info("👨 Left Robot: MALE voice (echo) - Curious, enthusiastic")
    logger.info("👩 Right Robot: FEMALE voice (shimmer) - Analytical, thoughtful")
    logger.info("🌊 Positioned waveforms based on speaking robot")
    logger.info("🔇 No chat simulator - pure robot conversations")

    streamer = GenderedVoiceRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 