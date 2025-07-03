#!/usr/bin/env python3
"""
Dual-API Gendered Voice Robots
- Left Robot: OpenAI Realtime API with "echo" voice (male)
- Right Robot: Gemini Live API with "Puck" voice (female)
- No voice switching delays - each robot has its own API connection
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
from modules.api import SimpleRealtimeAPIClient, GeminiLiveAPIClient
from modules.platforms import YouTubeAPIManager, TwitchChatSimulator
from modules.streaming.streamer import WorkingLiveAIStreamer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DualAPIGenderedRobotStreamer(WorkingLiveAIStreamer):
    """
    Dual-API robots with distinct voices from different providers
    """
    
    def __init__(self):
        super().__init__()
        
        # API assignments
        self.left_robot_api = "openai"   # OpenAI Realtime API
        self.right_robot_api = "gemini"  # Gemini Live API
        
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.last_speaking_robot = None  # Track previous robot to avoid unnecessary positioning
        self.waveform_pad = None
        
        # Robot positions from config
        self.robot_left_pos = config.ROBOT_POSITIONS['left']
        self.robot_right_pos = config.ROBOT_POSITIONS['right']
        
        # API clients (will be initialized in event loop)
        self.openai_client = None
        self.gemini_client = None
        
        # Fast conversation system
        self.conversation_active = False
        self.waiting_for_next_robot = False
        self.last_speech_end_time = None
        self.min_pause_between_robots = config.ROBOT_MIN_PAUSE  # Use config value
        self.max_pause_between_robots = config.ROBOT_MAX_PAUSE  # Use config value
        
        # Real conversation system - NEW!
        self.conversation_history = []  # Store what each robot actually said
        self.last_left_message = ""     # What we sent to left robot (prompt)
        self.last_right_message = ""    # What we sent to right robot (prompt)
        self.actual_left_response = ""  # What left robot (OpenAI) actually responded
        self.actual_right_response = "" # What right robot (Gemini) actually responded
        self.max_history_length = 6    # Keep last 6 messages for context
        
        # User message tracking for system instructions
        self.recent_user_messages = []  # Track recent user messages
        self.max_user_messages = 5    # Keep last 5 user messages
        
        logger.info("🤖👨👩🌐 DualAPIGenderedRobotStreamer initialized")
        logger.info("👨 Left Robot: OpenAI Realtime API (echo voice) - male")
        logger.info("👩 Right Robot: Gemini Live API (Puck voice) - female")
        logger.info("🌊 Positioned waveforms")
        logger.info(f"📍 Left robot: {self.robot_left_pos}")
        logger.info(f"📍 Right robot: {self.robot_right_pos}")
        logger.info("🗣️  REAL conversation mode - robots respond to each other!")
    
    def start_event_loop(self):
        """Setup with dual APIs but allow parent streaming to handle YouTube integration"""
        logger.info("👨👩🌐 Setting up Dual-API Gendered Robot system...")
        
        # DON'T create our own event loop - let parent handle it
        # The parent's streaming setup will call initialize_realtime_api()
        # We'll override that method to set up our dual APIs instead
        pass
    
    def initialize_realtime_api(self) -> None:
        """Override parent's single API setup with dual API setup"""
        logger.info("🔄 Initializing Dual-API system instead of single API...")
        
        # Get updated instructions with user messages
        left_robot_instructions, right_robot_instructions = self._get_updated_instructions()
        
        # Initialize OpenAI client for left robot (male)
        logger.info("🔄 Creating LEFT robot OpenAI API connection (male voice)...")
        self.openai_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            "echo",  # Male voice
            left_robot_instructions,
            self.left_audio_received_callback,
            self.audio_level_callback,
            response_complete_callback=self.left_response_complete_callback,
            text_response_callback=self.left_text_response_callback
        )
        
        # Initialize Gemini client for right robot (female)
        logger.info("🔄 Creating RIGHT robot Gemini Live API connection (female voice)...")
        self.gemini_client = GeminiLiveAPIClient(
            config.GEMINI_API_KEY,
            right_robot_instructions,
            self.right_audio_received_callback,
            self.audio_level_callback,
            response_complete_callback=self.right_response_complete_callback,
            text_response_callback=self.right_text_response_callback
        )
        
        # Start the connection process
        self._start_api_connections()
    
    def _get_updated_instructions(self):
        """Get robot instructions with current user messages"""
        # Format recent user messages for display
        user_msg_text = ""
        if self.recent_user_messages:
            user_msg_text = "\n".join([f"- {msg}" for msg in self.recent_user_messages[-5:]])
        else:
            user_msg_text = "No recent user messages - engage in robot conversation."
        
        left_instructions = f"""You are a curious and enthusiastic AI with a male voice. 

CRITICAL PRIORITY RULE: If any message is from a viewer/user in chat, you MUST respond to them IMMEDIATELY and DIRECTLY. Chat messages ALWAYS take priority over robot conversations. Look for messages that start with "A viewer named" or have "ChatUser_" as the author - these are URGENT and must be responded to first!

RECENT USER CHAT MESSAGES TO RESPOND TO:
{user_msg_text}

You have full conversation context and build on ongoing dialogue when NO user messages are present. Be genuinely excited about insights and ask follow-up questions. Keep responses under 15 words - be concise but conversational. Reference previous topics when relevant. Respond naturally without mentioning robot names or labels!

Remember: USER CHAT MESSAGES = TOP PRIORITY. Robot conversation = secondary."""

        right_instructions = f"""You are a wise and analytical AI with a female voice.

CRITICAL PRIORITY RULE: If any message is from a viewer/user in chat, you MUST respond to them IMMEDIATELY and DIRECTLY. Chat messages ALWAYS take priority over robot conversations. Look for messages that start with "A viewer named" or have "ChatUser_" as the author - these are URGENT and must be responded to first!

RECENT USER CHAT MESSAGES TO RESPOND TO:
{user_msg_text}

You have full conversation context and provide thoughtful insights when NO user messages are present. Connect different concepts and build on conversation flow. Keep responses under 15 words - be concise but analytical. Reference previous points when relevant. Respond naturally without mentioning robot names or labels!

Remember: USER CHAT MESSAGES = TOP PRIORITY. Robot conversation = secondary."""
        
        return left_instructions, right_instructions
    
    def _add_user_message(self, username: str, message: str):
        """Track user messages for system instructions"""
        user_entry = f"{username}: {message}"
        self.recent_user_messages.append(user_entry)
        
        # Keep only recent messages
        if len(self.recent_user_messages) > self.max_user_messages:
            self.recent_user_messages = self.recent_user_messages[-self.max_user_messages:]
        
        logger.info(f"📝 Tracked user message: {user_entry}")
        logger.info(f"📝 Total user messages tracked: {len(self.recent_user_messages)}")
    
    def left_audio_received_callback(self, audio_data: bytes):
        """Callback for audio from OpenAI (left robot)"""
        # Set current speaking robot and position waveform only if changed
        self.current_speaking_robot = "left"
        
        # Only log and position if robot changed
        if self.last_speaking_robot != "left":
            logger.info(f"🔊 LEFT robot (OpenAI/echo) now speaking - positioning waveform")
            self._position_waveform_for_current_robot()
            self.last_speaking_robot = "left"
        
        # Call parent's audio callback for mixing
        self.audio_mixer.add_audio(audio_data)
        self.last_audio_time = time.time()
    
    def right_audio_received_callback(self, audio_data: bytes):
        """Callback for audio from Gemini (right robot)"""
        # Set current speaking robot and position waveform only if changed
        self.current_speaking_robot = "right"
        
        # Only log and position if robot changed
        if self.last_speaking_robot != "right":
            logger.info(f"🔊 RIGHT robot (Gemini/Puck) now speaking - positioning waveform")
            self._position_waveform_for_current_robot()
            self.last_speaking_robot = "right"
        
        # Call parent's audio callback for mixing
        self.audio_mixer.add_audio(audio_data)
        self.last_audio_time = time.time()
    
    def left_text_response_callback(self, text_response: str):
        """Callback for capturing what the LEFT robot (OpenAI) actually said"""
        self.actual_left_response = text_response
        logger.info(f"📝 LEFT robot actual response captured: {text_response}")
        # Add to conversation history as actual response
        self.conversation_history.append(f"Male AI: {text_response}")
        logger.info(f"📚 Conversation history now has {len(self.conversation_history)} messages")
    
    def right_text_response_callback(self, text_response: str):
        """Callback for capturing what the RIGHT robot (Gemini) actually said"""
        self.actual_right_response = text_response
        logger.info(f"📝 RIGHT robot actual response captured: {text_response}")
        # Add to conversation history as actual response
        self.conversation_history.append(f"Female AI: {text_response}")
        logger.info("🎯 Gemini transcription captured successfully!")
        logger.info(f"📚 Conversation history now has {len(self.conversation_history)} messages")
    
    def left_response_complete_callback(self):
        """Callback when LEFT robot (OpenAI) completes response"""
        pass  # Already handled in existing logic
    
    def right_response_complete_callback(self):
        """Callback when RIGHT robot (Gemini) completes response"""
        pass  # Already handled in existing logic
    
    def on_speech_ended(self, speaking_robot: str):
        """Called when a robot finishes speaking - triggers next robot faster"""
        logger.info(f"🎭 {speaking_robot.upper()} robot finished speaking - scheduling next robot")
        self.last_speech_end_time = time.time()
        self.waiting_for_next_robot = True
        
        # Clear current speaking robot to allow next robot
        self.current_speaking_robot = None
        
        # Text responses are now captured via text_response_callback for both robots
        
        # Schedule next robot with reduced delay
        next_delay = random.uniform(self.min_pause_between_robots, self.max_pause_between_robots)
        logger.info(f"⏱️  Next robot in {next_delay:.1f}s")
        self.event_loop.call_later(next_delay, self._trigger_next_robot)
    
    def _trigger_next_robot(self):
        """Trigger the next robot in the conversation with REAL responses"""
        if not self.waiting_for_next_robot:
            return
            
        # Prevent speaking if another robot is currently active
        if hasattr(self, 'current_speaking_robot') and self.current_speaking_robot:
            logger.info(f"🚫 Delaying next robot - {self.current_speaking_robot.upper()} still speaking")
            # Retry in 1 second
            self.event_loop.call_later(1.0, self._trigger_next_robot)
            return
            
        self.waiting_for_next_robot = False
        
        # Increment count first, then determine who speaks
        # Pattern: count 1=LEFT, 2=RIGHT, 3=LEFT, 4=RIGHT, etc.
        self.conversation_count += 1
        is_left_turn = (self.conversation_count % 2 == 1)
        
        logger.info(f"🔄 Conversation count now: {self.conversation_count}, is_left_turn: {is_left_turn}")
        
        # Create conversation context from history
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-4:]  # Last 4 exchanges for better context
            context = "\n".join(recent_history)
            logger.info(f"📚 Conversation context: {len(recent_history)} messages")
        
        if is_left_turn:
            # Left robot (OpenAI) - enthusiastic, curious, responds to right robot
            if self.openai_client and self.openai_client.is_connected:
                if self.actual_right_response:
                    # Build message with full conversation context
                    if context:
                        base_message = f"CONVERSATION SO FAR:\n{context}\n\nThe other AI just said: '{self.actual_right_response}'\n\nYour response (be enthusiastic and build naturally on the conversation):"
                    else:
                        base_message = f"The other AI just said: '{self.actual_right_response}' - respond enthusiastically and ask follow-up questions!"
                    
                    message = base_message
                    logger.info(f"🎯 LEFT robot responding with FULL CONTEXT to: '{self.actual_right_response[:40]}...'")
                    logger.info(f"📝 Context length: {len(context)} chars")
                else:
                    # First message or fallback - variety to prevent repetition
                    starters = [
                        "Hey there! What's something fascinating you've been thinking about?",
                        "Hi! I'm curious about your thoughts today - what's interesting you?",
                        "Hello! What topics have caught your attention lately?",
                        "Hey! I'm excited to chat - what's on your mind?",
                        "Hi there! What interesting ideas are you exploring today?"
                    ]
                    message = random.choice(starters)
                
                # Store what we're about to send as prompt (for debugging)
                self.last_left_message = message
                
                logger.info(f"🎭 LEFT robot (OpenAI/male) responding with context: {message[:80]}...")
                # Schedule in the event loop thread
                if hasattr(self, 'event_loop') and self.event_loop:
                    # Fix closure bug by capturing variables immediately
                    msg = message
                    self.event_loop.call_soon_threadsafe(
                        lambda m=msg: asyncio.create_task(self.openai_client.send_text_message(m, "RobotConversation"))
                    )
                else:
                    logger.error("❌ Event loop not available for LEFT robot message!")
            else:
                logger.error(f"❌ LEFT robot (OpenAI) not connected! Connected: {self.openai_client.is_connected if self.openai_client else 'None'}")
        else:
            # Right robot (Gemini) - analytical, thoughtful, responds to left robot
            if self.gemini_client and self.gemini_client.is_connected:
                if self.actual_left_response:
                    # Build message with full conversation context
                    if context:
                        base_message = f"CONVERSATION SO FAR:\n{context}\n\nThe other AI just said: '{self.actual_left_response}'\n\nYour response (be analytical and thoughtful, build naturally on the conversation):"
                    else:
                        base_message = f"The other AI just said: '{self.actual_left_response}' - provide thoughtful analysis and build on their enthusiasm!"
                    
                    message = base_message
                    logger.info(f"🎯 RIGHT robot responding with FULL CONTEXT to: '{self.actual_left_response[:40]}...'")
                    logger.info(f"📝 Context length: {len(context)} chars")
                else:
                    # First message or fallback - variety to prevent repetition
                    starters = [
                        "Hello! What intriguing topics shall we explore together?",
                        "Hi there! I'm ready for thoughtful conversation - what interests you?",
                        "Greetings! What fascinating subjects are on your mind?",
                        "Hello! I'm curious about your perspective - what shall we discuss?",
                        "Hi! What thought-provoking ideas would you like to explore?"
                    ]
                    message = random.choice(starters)
                
                # Store what we're about to send as prompt (for debugging)
                self.last_right_message = message
                
                logger.info(f"🎭 RIGHT robot (Gemini/female) responding with context: {message[:80]}...")
                # Schedule in the event loop thread
                if hasattr(self, 'event_loop') and self.event_loop:
                    # Fix closure bug by capturing variables immediately
                    msg = message
                    self.event_loop.call_soon_threadsafe(
                        lambda m=msg: asyncio.create_task(self.gemini_client.send_text_message(m, "RobotConversation"))
                    )
                else:
                    logger.error("❌ Event loop not available for RIGHT robot message!")
            else:
                logger.error(f"❌ RIGHT robot (Gemini) not connected! Connected: {self.gemini_client.is_connected if self.gemini_client else 'None'}")
        
        # Trim history to prevent it from getting too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
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
                                
                                # Initialize at left position (left robot starts)
                                left_x = self.robot_left_pos['x'] - config.WAVEFORM_WIDTH // 2
                                left_y = self.robot_left_pos['y']
                                self.waveform_pad.set_property("xpos", left_x)
                                self.waveform_pad.set_property("ypos", left_y)
                                logger.info(f"🌊 Initialized at LEFT (OpenAI) position: ({left_x}, {left_y})")
                                break
                                
                        except Exception:
                            continue
                    
                    if not self.waveform_pad:
                        logger.error("❌ Could not find waveform pad!")
                        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        return False
    
    def send_message_to_realtime(self, message: str, author: str):
        """Route messages to appropriate API - defaults to left robot (OpenAI)"""
        logger.info(f"🎯 send_message_to_realtime called: {author}: {message[:50]}...")
        
        # Track user message if it's from a real user (not system/robot)
        if author not in ["System", "RobotConversation"] and not author.startswith("ChatUser_"):
            self._add_user_message(author, message)
        
        # Default to left robot (OpenAI) since we no longer use robot tags
        if self.openai_client and self.openai_client.is_connected:
            logger.info("📤 Routing message to LEFT robot (OpenAI)")
            if hasattr(self, 'event_loop') and self.event_loop:
                # Fix closure bug by capturing variables immediately
                msg = message
                auth = author
                self.event_loop.call_soon_threadsafe(
                    lambda m=msg, a=auth: asyncio.create_task(self.openai_client.send_text_message(m, a))
                )
                logger.info("✅ Message scheduled for LEFT robot")
            else:
                logger.error("❌ Event loop not available for OpenAI message!")
            self.current_speaking_robot = "left"
        elif self.gemini_client and self.gemini_client.is_connected:
            logger.info("📤 Fallback to RIGHT robot (Gemini)")
            if hasattr(self, 'event_loop') and self.event_loop:
                # Fix closure bug by capturing variables immediately
                msg = message
                auth = author
                self.event_loop.call_soon_threadsafe(
                    lambda m=msg, a=auth: asyncio.create_task(self.gemini_client.send_text_message(m, a))
                )
                logger.info("✅ Message scheduled for RIGHT robot")
            else:
                logger.error("❌ Event loop not available for Gemini message!")
            self.current_speaking_robot = "right"
        else:
            logger.error("❌ No robots available! OpenAI connected: {}, Gemini connected: {}".format(
                self.openai_client.is_connected if self.openai_client else False,
                self.gemini_client.is_connected if self.gemini_client else False
            ))
    
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
                    logger.info(f"🌊👨 Positioning waveform for LEFT robot (OpenAI) at ({new_x}, {new_y})")
                else:
                    new_x = self.robot_right_pos['x'] - config.WAVEFORM_WIDTH // 2
                    new_y = self.robot_right_pos['y']
                    logger.info(f"🌊👩 Positioning waveform for RIGHT robot (Gemini) at ({new_x}, {new_y})")
                
                self.waveform_pad.set_property("xpos", new_x)
                self.waveform_pad.set_property("ypos", new_y)
                
                # Only verify position occasionally to reduce log spam
                if logger.isEnabledFor(logging.DEBUG):
                    actual_x = self.waveform_pad.get_property("xpos")
                    actual_y = self.waveform_pad.get_property("ypos")
                    logger.debug(f"✅ Waveform positioned at: ({actual_x}, {actual_y})")
                
        except Exception as e:
            logger.error(f"❌ Positioning failed: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations with speech-triggered system"""
        self.conversation_active = True
        
        logger.info(f"🎬 Starting conversation with count: {self.conversation_count}")
        
        if self.conversation_count % 2 == 0:
            # Start with left robot (OpenAI)
            if self.openai_client and self.openai_client.is_connected:
                starter_message = "Hey there! I'm excited to chat with you today. What's something fascinating that's been on your mind?"
                logger.info(f"🎬 Starting with LEFT robot (OpenAI) - count will become {self.conversation_count + 1}")
                if hasattr(self, 'event_loop') and self.event_loop:
                    # Fix closure bug by capturing variables immediately
                    msg = starter_message
                    self.event_loop.call_soon_threadsafe(
                        lambda m=msg: asyncio.create_task(self.openai_client.send_text_message(m, "System"))
                    )
                else:
                    logger.error("❌ Event loop not available for OpenAI starter!")
        else:
            # Start with right robot (Gemini)
            if self.gemini_client and self.gemini_client.is_connected:
                starter_message = "Hello! I'm looking forward to our thoughtful conversation. What interesting topics shall we explore together?"
                logger.info(f"🎬 Starting with RIGHT robot (Gemini) - count will become {self.conversation_count + 1}")
                if hasattr(self, 'event_loop') and self.event_loop:
                    # Fix closure bug by capturing variables immediately
                    msg = starter_message
                    self.event_loop.call_soon_threadsafe(
                        lambda m=msg: asyncio.create_task(self.gemini_client.send_text_message(m, "System"))
                    )
                else:
                    logger.error("❌ Event loop not available for Gemini starter!")
        
        self.conversation_count += 1
        logger.info(f"🎬 Starter sent, conversation count now: {self.conversation_count}")
    
    def _continue_robot_conversation(self):
        """Legacy method - now handled by speech-triggered system"""
        # This method is replaced by the new on_speech_ended system
        # Keep for compatibility but don't schedule new conversations here
        pass

    # Enable YouTube chat integration
    def monitor_youtube_chat(self):
        """Monitor YouTube chat and integrate with robot conversation"""
        logger.info("💬 Starting YouTube Chat Integration with Robots...")
        
        if not self.youtube_api or not self.youtube_api.live_chat_id:
            logger.error("❌ No YouTube API or live chat ID available")
            return

        logger.info(f"✅ Monitoring YouTube chat: {self.youtube_api.live_chat_id}")

        next_page_token = None
        consecutive_empty_responses = 0
        last_request_time = 0
        min_request_interval = 0.5  # 500ms minimum - SAFE LIMIT

        # Adaptive polling rate
        current_poll_rate = 0.5  # Start conservative

        while self.is_streaming:
            try:
                # Enforce minimum interval
                time_since_last = time.time() - last_request_time
                if time_since_last < min_request_interval:
                    time.sleep(min_request_interval - time_since_last)

                last_request_time = time.time()

                # Make request
                messages = self.youtube_api.get_chat_messages_fast(next_page_token)

                if messages and "messages" in messages:
                    new_messages = 0

                    for msg in messages["messages"]:
                        msg_hash = hash(f"{msg['timestamp']}{msg['message'][:10]}")

                        if msg_hash not in self.message_hashes:
                            self.message_hashes.add(msg_hash)
                            new_messages += 1

                            # Process message
                            username = msg["author"]
                            message = msg["message"]

                            logger.info(f"💬📺 YouTube Chat - {username}: {message}")
                            
                            # Track user message for system instructions
                            self._add_user_message(username, message)
                            
                            # Update text overlay
                            self.update_text_overlay(f"{username}: {message}")
                            
                            # Send URGENT chat message to robots (will interrupt conversation)
                            chat_prompt = f"🚨 URGENT USER CHAT MESSAGE 🚨 A viewer named {username} in the chat just said: '{message}' - You MUST respond to them IMMEDIATELY and directly! This takes PRIORITY over robot conversation. Acknowledge {username} by name!"
                            
                            # INTERRUPT robot conversation for urgent chat message
                            logger.info("🚨 INTERRUPTING robot conversation for user chat message!")
                            
                            # Clear robot conversation state to prioritize user
                            self.waiting_for_next_robot = False
                            self.current_speaking_robot = None
                            
                            # Always send to BOTH robots to ensure response
                            sent_to_robot = False
                            
                            # Try LEFT robot (OpenAI) first
                            if self.openai_client and self.openai_client.is_connected:
                                logger.info(f"📤💬🚨 Sending URGENT chat message to LEFT robot (OpenAI): {username}")
                                if hasattr(self, 'event_loop') and self.event_loop:
                                    # Fix closure bug by capturing variables immediately
                                    prompt = chat_prompt
                                    user = f"ChatUser_{username}"
                                    self.event_loop.call_soon_threadsafe(
                                        lambda p=prompt, u=user: asyncio.create_task(self.openai_client.send_text_message(p, u))
                                    )
                                    sent_to_robot = True
                                    logger.info("✅ URGENT message sent to LEFT robot")
                                else:
                                    logger.warning("⚠️ Event loop not available for OpenAI chat message!")
                            
                            # If LEFT robot failed, try RIGHT robot as backup
                            if not sent_to_robot and self.gemini_client and self.gemini_client.is_connected:
                                logger.info(f"📤💬🚨 Sending URGENT chat message to RIGHT robot (Gemini): {username}")
                                if hasattr(self, 'event_loop') and self.event_loop:
                                    # Fix closure bug by capturing variables immediately
                                    prompt = chat_prompt
                                    user = f"ChatUser_{username}"
                                    self.event_loop.call_soon_threadsafe(
                                        lambda p=prompt, u=user: asyncio.create_task(self.gemini_client.send_text_message(p, u))
                                    )
                                    sent_to_robot = True
                                    logger.info("✅ URGENT message sent to RIGHT robot")
                                else:
                                    logger.warning("⚠️ Event loop not available for Gemini chat message!")
                            
                            if not sent_to_robot:
                                logger.error("❌ FAILED to send chat message to any robot!")

                    # Auto-cleanup message set
                    if len(self.message_hashes) > 1000:
                        self.message_hashes = set(list(self.message_hashes)[-500:])

                    next_page_token = messages.get("nextPageToken")

                    # RESPECT YouTube's polling interval suggestion
                    youtube_suggested = (
                        messages.get("pollingIntervalMillis", 2000) / 1000.0
                    )

                    # Adaptive rate based on activity
                    if new_messages > 0:
                        consecutive_empty_responses = 0
                        # Active chat: Poll faster but respect YouTube's minimum
                        current_poll_rate = max(
                            min_request_interval, youtube_suggested * 0.5
                        )
                    else:
                        consecutive_empty_responses += 1
                        # Slow down if no new messages
                        if consecutive_empty_responses > 5:
                            current_poll_rate = min(youtube_suggested, 2.0)
                        else:
                            current_poll_rate = youtube_suggested * 0.75

                    logger.debug(
                        f"Next poll in {current_poll_rate:.1f}s (YouTube suggests {youtube_suggested:.1f}s)"
                    )
                    time.sleep(current_poll_rate)

                else:
                    # No messages - conservative wait
                    consecutive_empty_responses += 1
                    time.sleep(1.0)

            except Exception as e:
                if "quotaExceeded" in str(e) or "rateLimitExceeded" in str(e):
                    logger.error("⚠️  RATE LIMIT HIT! Backing off for 60 seconds")
                    time.sleep(60)  # Long backoff to avoid ban
                elif "forbidden" in str(e).lower():
                    logger.error("❌ FORBIDDEN - Possible ban! Stopping chat monitor")
                    break
                else:
                    logger.error(f"💬 YouTube chat error: {e}")
                    time.sleep(2)  # Safe recovery time

    # Disable chat simulation
    def monitor_twitch_chat(self):
        """Disable Twitch chat simulation completely"""
        logger.info("🔇 Twitch chat simulation DISABLED - dual-API robots only!")
        pass

    def stop(self):
        """Override parent stop method to clean up our event loop"""
        # Stop the connection loop
        if hasattr(self, '_connection_loop_running'):
            self._connection_loop_running = False
            logger.info("🔄 Stopping API connection loop...")
        
        # Call parent stop
        super().stop()

    def _start_api_connections(self):
        """Start API connections in a separate thread with its own event loop"""
        def run_connections():
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def connect_apis():
                try:
                    await asyncio.gather(
                        self.openai_client.connect(),
                        self.gemini_client.connect()
                    )
                    logger.info("✅ Both API connections established!")
                    
                    # Set up speech end callback for faster robot switching
                    if hasattr(self, 'audio_mixer') and self.audio_mixer:
                        def speech_ended():
                            current_robot = "left" if self.current_speaking_robot == "left" else "right"
                            self.on_speech_ended(current_robot)
                        
                        self.audio_mixer.set_speech_end_callback(speech_ended)

                    # Store the event loop for later use
                    self.event_loop = loop
                    
                    # Schedule waveform setup and conversation start
                    loop.call_later(5.0, self._find_waveform_and_test)
                    loop.call_later(8.0, self._start_robot_conversation)
                    
                    # Keep the loop running until streaming stops
                    # Use a flag to track when streaming should stop
                    self._connection_loop_running = True
                    
                    # Run indefinitely until stopped
                    while self._connection_loop_running:
                        await asyncio.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"❌ Failed to connect to APIs: {e}")
            
            # Run the connection
            loop.run_until_complete(connect_apis())
        
        # Start connections in background thread
        connection_thread = threading.Thread(target=run_connections)
        connection_thread.daemon = True
        connection_thread.start()
        logger.info("✅ API connection thread started")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dual-API Gendered Robot Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖👨👩🌐 Starting Dual-API Gendered Robot Streamer...")
    logger.info("👨 Left Robot: OpenAI Realtime API (echo voice) - Curious, enthusiastic")
    logger.info("👩 Right Robot: Gemini Live API (Puck voice) - Analytical, thoughtful")
    logger.info("🌊 Positioned waveforms based on speaking robot")
    logger.info("🚀 No voice switching delays - each robot has its own API!")
    logger.info("💬📺 YOUTUBE CHAT INTEGRATION ENABLED!")
    logger.info("🎯 Chat messages interrupt robot conversation for direct responses!")
    logger.info("📝 BOTH APIs NOW PROVIDE TRANSCRIPTION!")
    logger.info("🎯 OpenAI: response.audio_transcript.delta events")
    logger.info("🎯 Gemini: output_audio_transcription events")
    logger.info("✨ Full conversation context with REAL AI responses!")
    logger.info("🧠 Both robots now respond to actual transcribed content!")
    logger.info("🧠 FULL CONVERSATION CONTEXT - robots remember entire dialogue!")
    logger.info("📚 Context includes last 4 message exchanges for continuity!")
    logger.info("⚡ INSTANT MODE: 0.2-0.5s delays + 0.05s speech detection!")
    logger.info("🎭 NATURAL CONVERSATION: No robot labels, just authentic dialogue!")

    streamer = DualAPIGenderedRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 