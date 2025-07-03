"""
OpenAI Realtime API WebSocket Client
"""

import json
import time
import asyncio
import websockets
import base64
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class SimpleRealtimeAPIClient:
    """Simplified OpenAI Realtime API WebSocket Client with audio level callback"""

    def __init__(
        self, api_key: str, realtime_url: str, model: str, voice: str, 
        instructions: str, audio_callback: Callable, level_callback: Callable = None,
        response_complete_callback: Callable = None, text_response_callback: Callable = None
    ):
        self.api_key = api_key
        self.realtime_url = realtime_url
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.ws = None
        self.audio_callback = audio_callback
        self.level_callback = level_callback
        self.response_complete_callback = response_complete_callback
        self.text_response_callback = text_response_callback
        self.current_text_response = ""  # Track text response for capture
        self.is_connected = False
        self.is_speaking = False
        self.last_audio_time = None
        self.current_text_response = ""

    async def connect(self):
        """Connect to OpenAI Realtime API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            url = f"{self.realtime_url}?model={self.model}"

            self.ws = await websockets.connect(
                url, additional_headers=headers, ping_interval=20, ping_timeout=10
            )

            self.is_connected = True
            logger.info("✅ Connected to OpenAI Realtime API")

            # Configure session
            await self.configure_session()

            # Start listening
            asyncio.create_task(self.listen())

        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            self.is_connected = False

    async def configure_session(self):
        """Configure session with optimized settings"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "temperature": 0.7,
                "max_response_output_tokens": 2048,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                },
            },
        }

        await self.ws.send(json.dumps(config))
        logger.info("✅ Session configured for smooth audio")
        await asyncio.sleep(0.5)

    async def listen(self):
        """Listen for messages"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                event_type = data.get("type")

                if event_type == "response.audio.delta":
                    audio_data = base64.b64decode(data.get("delta", ""))
                    if audio_data:
                        timestamp = time.time()
                        if self.last_audio_time:
                            delta = timestamp - self.last_audio_time
                            logger.debug(
                                f"[OPENAI] Audio chunk: {len(audio_data)} bytes, delta: {delta:.3f}s"
                            )
                        else:
                            logger.info("[OPENAI] Audio stream started")
                        self.last_audio_time = timestamp

                        # Send audio to callback
                        if self.audio_callback:
                            self.audio_callback(audio_data)

                        # Send audio levels for visualization
                        if self.level_callback:
                            self.level_callback(audio_data)

                        self.is_speaking = True

                elif event_type == "response.audio.done":
                    logger.info("[OPENAI] Audio response completed")
                    self.is_speaking = False
                    self.last_audio_time = None

                    # Send silence to level callback to fade out waveform
                    if self.level_callback:
                        silence = bytes(2048)
                        for _ in range(
                            5
                        ):  # Send multiple silence chunks for smooth fade
                            self.level_callback(silence)
                    
                    # Audio response complete - text transcription handled in response.audio_transcript.done
                    logger.info("[OPENAI] 🎵 Audio response complete")
                    
                    # Notify that response is complete
                    if self.response_complete_callback:
                        self.response_complete_callback()

                elif event_type == "response.text.delta":
                    # This is for input transcription (user speech), not AI responses
                    text_delta = data.get("delta", "")
                    logger.debug(f"[OPENAI] Input transcription delta: {text_delta}")

                elif event_type == "response.audio_transcript.delta":
                    # This is the AI's speech transcription that we need!
                    text_delta = data.get("delta", "")
                    self.current_text_response += text_delta
                    logger.info(f"[OPENAI] 📝 AI speech transcribed: {text_delta}")

                elif event_type == "response.audio_transcript.done":
                    # Full AI speech transcript is complete
                    if self.text_response_callback and self.current_text_response.strip():
                        logger.info(f"[OPENAI] 📝 Captured text response: {self.current_text_response[:100]}...")
                        self.text_response_callback(self.current_text_response.strip())
                        self.current_text_response = ""  # Reset for next response

                elif event_type == "session.updated":
                    session = data.get("session", {})
                    voice = session.get("voice", "unknown")
                    logger.info(f"[OPENAI] ✅ Session updated - Voice confirmed: {voice}")

                elif event_type == "error":
                    error = data.get("error", {})
                    error_message = error.get("message", "")
                    if "Cannot update a conversation's voice if assistant audio is present" in error_message:
                        logger.warning(f"[OPENAI] 🔄 Voice change blocked - assistant audio still present, will retry")
                    else:
                        logger.error(f"[OPENAI] Realtime API error: {error}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self.is_connected = False

    async def send_text_message(self, text: str, author: str):
        """Send text message for audio response"""
        if not self.is_connected or not self.ws:
            logger.error("Not connected to Realtime API")
            return

        try:
            logger.info(f"Sending to Realtime API - {author}: {text}")

            # Create conversation item
            item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{author} says: {text}"}
                    ],
                },
            }

            await self.ws.send(json.dumps(item))

            # Trigger response
            response = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": self.instructions,
                },
            }

            await self.ws.send(json.dumps(response))
            logger.info("Response request sent")

        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def send_session_update(self, session_updates: dict):
        """Send session update to change voice or other settings"""
        if not self.is_connected or not self.ws:
            logger.error("[OPENAI] Not connected to Realtime API")
            return
        
        try:
            # Update local voice if provided
            old_voice = self.voice
            if "voice" in session_updates:
                self.voice = session_updates["voice"]
                logger.info(f"[OPENAI] 🎤 Voice change: '{old_voice}' → '{self.voice}'")
            
            # Create session update message
            session_update = {
                "type": "session.update", 
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self.instructions,
                    "voice": self.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": 0.7,
                    "max_response_output_tokens": 2048,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                    },
                    **session_updates  # Override with any provided updates
                }
            }
            
            logger.info(f"[OPENAI] 📤 Sending session update with voice: {self.voice}")
            await self.ws.send(json.dumps(session_update))
            logger.info(f"[OPENAI] ✅ Session update sent successfully")
            
        except Exception as e:
            logger.error(f"[OPENAI] ❌ Error sending session update: {e}") 