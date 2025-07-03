"""
Dual Robot OpenAI Realtime API Manager
Manages conversations between two robots and incorporates user messages
"""

import json
import time
import asyncio
import threading
import logging
import os
from typing import List, Dict, Any, Callable
from .openai_realtime import SimpleRealtimeAPIClient
from .. import config

logger = logging.getLogger(__name__)


class UserMessageStorage:
    """Handles storage and retrieval of user chat messages"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.messages = []
        self.load_messages()
    
    def load_messages(self):
        """Load messages from file"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
        except Exception as e:
            logger.error(f"Error loading user messages: {e}")
            self.messages = []
    
    def save_messages(self):
        """Save messages to file"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving user messages: {e}")
    
    def add_message(self, author: str, message: str):
        """Add a new user message"""
        message_data = {
            "author": author,
            "message": message,
            "timestamp": time.time()
        }
        self.messages.append(message_data)
        
        # Keep only the most recent messages
        if len(self.messages) > config.MAX_STORED_USER_MESSAGES:
            self.messages = self.messages[-config.MAX_STORED_USER_MESSAGES:]
        
        self.save_messages()
        logger.info(f"Stored user message from {author}: {message}")
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent user messages"""
        return self.messages[-count:] if self.messages else []
    
    def get_messages_summary(self, count: int = 10) -> str:
        """Get a summary of recent user messages for robot context"""
        recent = self.get_recent_messages(count)
        if not recent:
            return "No recent user messages."
        
        summary = "Recent user comments: "
        for msg in recent:
            summary += f"{msg['author']}: {msg['message']} | "
        return summary.rstrip("| ")


class DualRobotRealtimeManager:
    """Manages conversations between two robots using OpenAI Realtime API"""
    
    def __init__(self, left_audio_callback: Callable, right_audio_callback: Callable,
                 left_level_callback: Callable = None, right_level_callback: Callable = None):
        self.left_client = None
        self.right_client = None
        self.left_audio_callback = left_audio_callback
        self.right_audio_callback = right_audio_callback
        self.left_level_callback = left_level_callback
        self.right_level_callback = right_level_callback
        
        # Conversation state
        self.conversation_active = False
        self.current_speaker = None  # 'left' or 'right'
        self.conversation_turn = 0
        self.conversation_history = []
        self.last_response_time = 0
        
        # User message storage
        self.user_storage = UserMessageStorage(config.USER_MESSAGES_FILE)
        
        # Event loop management
        self.event_loop = None
        self.loop_thread = None
        
        # Conversation control
        self.conversation_task = None
        self.waiting_for_response = False
    
    def start_event_loop(self):
        """Start the asyncio event loop in a separate thread"""
        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Create robot clients
            self.left_client = SimpleRealtimeAPIClient(
                config.OPENAI_API_KEY,
                config.OPENAI_REALTIME_URL,
                config.OPENAI_REALTIME_MODEL,
                config.ROBOT_VOICES[0],  # "alloy" for left robot
                config.ROBOT_LEFT_SYSTEM_MESSAGE,
                self.left_audio_callback,
                self.left_level_callback,
                lambda: self.on_robot_response_complete('left')
            )
            
            self.right_client = SimpleRealtimeAPIClient(
                config.OPENAI_API_KEY,
                config.OPENAI_REALTIME_URL,
                config.OPENAI_REALTIME_MODEL,
                config.ROBOT_VOICES[1],  # "echo" for right robot
                config.ROBOT_RIGHT_SYSTEM_MESSAGE,
                self.right_audio_callback,
                self.right_level_callback,
                lambda: self.on_robot_response_complete('right')
            )
            
            # Connect both clients
            self.event_loop.run_until_complete(self._connect_robots())
            
            # Start conversation after a delay
            self.event_loop.call_later(5.0, self._start_conversation_async)
            
            # Keep the loop running
            self.event_loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()
        
        # Wait a moment for the event loop to start
        time.sleep(2)
    
    async def _connect_robots(self):
        """Connect both robot clients"""
        try:
            await asyncio.gather(
                self.left_client.connect(),
                self.right_client.connect()
            )
            logger.info("✅ Both robots connected successfully")
        except Exception as e:
            logger.error(f"Error connecting robots: {e}")
    
    def _start_conversation_async(self):
        """Start the robot conversation (async wrapper for synchronous call)"""
        if self.event_loop:
            asyncio.run_coroutine_threadsafe(self.start_robot_conversation(), self.event_loop)
    
    async def start_robot_conversation(self):
        """Start the conversation between robots"""
        if self.conversation_active:
            return
        
        self.conversation_active = True
        self.conversation_turn = 0
        self.current_speaker = 'left'  # Left robot starts
        
        logger.info("🤖 Starting robot conversation")
        
        # Initial greeting from left robot
        await self._send_message_to_robot('left', 
            "Hello there! I'm excited to chat with you today. What's on your mind?", 
            "System")
    
    async def _send_message_to_robot(self, robot: str, message: str, author: str):
        """Send a message to a specific robot"""
        client = self.left_client if robot == 'left' else self.right_client
        
        if client and client.is_connected:
            self.waiting_for_response = True
            self.current_speaker = robot
            
            # Store the conversation entry
            self.conversation_history.append({
                "speaker": robot,
                "message": message,
                "timestamp": time.time()
            })
            
            await client.send_text_message(message, author)
            logger.info(f"[DUAL_ROBOT] Sent message to {robot} robot: {message[:50]}...")
            
        else:
            logger.error(f"Robot {robot} client not connected")
    
    def _continue_conversation_async(self):
        """Continue the conversation (async wrapper)"""
        if self.event_loop and not self.waiting_for_response:
            asyncio.run_coroutine_threadsafe(self._continue_conversation(), self.event_loop)
    
    async def _continue_conversation(self):
        """Continue the robot conversation"""
        if not self.conversation_active or self.waiting_for_response:
            return
        
        self.conversation_turn += 1
        
        # Switch speakers
        next_speaker = 'right' if self.current_speaker == 'left' else 'left'
        
        # Prepare context for the next robot
        context = self._build_conversation_context()
        
        # Check if we should incorporate user messages
        if self.conversation_turn % config.ROBOT_MAX_CONVERSATION_TURNS == 0:
            user_summary = self.user_storage.get_messages_summary()
            if user_summary != "No recent user messages.":
                context += f" Also, here are some recent user comments to consider: {user_summary}"
        
        await self._send_message_to_robot(next_speaker, context, "Conversation Partner")
    
    def _build_conversation_context(self) -> str:
        """Build context for the next robot response"""
        if len(self.conversation_history) == 0:
            return "Let's have an interesting conversation! What would you like to discuss?"
        
        # Get the actual content from the last conversation, not the wrapped message
        if len(self.conversation_history) > 0:
            last_entry = self.conversation_history[-1]
            
            # Extract the actual robot message, not our system prompts
            last_message = last_entry['message']
            
            # If this was a system prompt to start conversation, generate a natural topic
            if "Hello there!" in last_message or "What's on your mind?" in last_message:
                topics = [
                    "I've been thinking about how fascinating human creativity is. What inspires you most?",
                    "Technology keeps evolving so quickly. What changes excite you?", 
                    "I'm curious about human relationships. How do you connect with others?",
                    "Learning never stops, does it? What's something new you discovered recently?",
                    "The world has so many interesting perspectives. What's your view on that?"
                ]
                import random
                return random.choice(topics)
            
            # For subsequent responses, respond naturally to the partner
            partner_name = "RoboChat-R" if self.current_speaker == 'left' else "RoboChat-L"
            
            # Don't repeat the whole conversation, just respond to the last meaningful exchange
            return f"Continue our conversation naturally. Keep it friendly and engaging, under 20 words."
        
        return "Please continue our conversation naturally."
    
    def add_user_message(self, author: str, message: str):
        """Add a user message to storage"""
        self.user_storage.add_message(author, message)
    
    def on_robot_response_complete(self, robot: str):
        """Called when a robot finishes responding"""
        self.waiting_for_response = False
        self.last_response_time = time.time()
        
        logger.info(f"Robot {robot} finished responding")
        
        # Schedule next response after delay
        if self.conversation_active and self.event_loop:
            self.event_loop.call_later(
                config.ROBOT_CONVERSATION_DELAY,
                self._continue_conversation_async
            )
    
    def stop_conversation(self):
        """Stop the robot conversation"""
        self.conversation_active = False
        if self.conversation_task:
            self.conversation_task.cancel()
        
        logger.info("🛑 Robot conversation stopped")
    
    def is_robot_speaking(self) -> tuple:
        """Returns (left_speaking, right_speaking)"""
        left_speaking = self.left_client.is_speaking if self.left_client else False
        right_speaking = self.right_client.is_speaking if self.right_client else False
        return (left_speaking, right_speaking)
