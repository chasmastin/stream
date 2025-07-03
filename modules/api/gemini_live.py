"""
Google Gemini Live API WebSocket Client
"""

import json
import time
import asyncio
import websockets
import base64
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class GeminiLiveAPIClient:
    """Google Gemini Live API WebSocket Client with audio support"""

    def __init__(
        self, api_key: str, instructions: str, audio_callback: Callable, 
        level_callback: Callable = None, response_complete_callback: Callable = None,
        text_response_callback: Callable = None
    ):
        self.api_key = api_key
        self.instructions = instructions
        self.ws = None
        self.audio_callback = audio_callback
        self.level_callback = level_callback
        self.response_complete_callback = response_complete_callback
        self.text_response_callback = text_response_callback
        self.is_connected = False
        self.is_speaking = False
        self.last_audio_time = None
        self.setup_id = None
        self.current_text_response = ""  # Track text response (for STT later)

    async def connect(self):
        """Connect to Gemini Live API"""
        try:
            # Gemini Live uses query parameter for API key
            url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"

            self.ws = await websockets.connect(
                url, ping_interval=20, ping_timeout=10
            )

            self.is_connected = True
            logger.info("✅ Connected to Gemini Live API")

            # Setup the session
            await self.setup_session()

            # Start listening
            asyncio.create_task(self.listen())

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live API: {e}")
            self.is_connected = False

    async def setup_session(self):
        """Setup Gemini Live session with transcription support"""
        setup_message = {
            "setup": {
                "model": "models/gemini-2.0-flash-live-001",
                "generation_config": {
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Puck"  # Female voice
                            }
                        }
                    },
                    "response_modalities": ["AUDIO"],  # Audio output for speech
                    "temperature": 0.7,
                    "max_output_tokens": 2048
                },
                "system_instruction": {
                    "parts": [
                        {
                            "text": self.instructions
                        }
                    ]
                },
                "tools": [],
                # ENABLE TRANSCRIPTION CAPTURE - This is the key addition!
                "input_audio_transcription": {},   # Capture user speech as text
                "output_audio_transcription": {}   # Capture AI speech as text
            }
        }

        await self.ws.send(json.dumps(setup_message))
        logger.info("✅ Gemini Live session setup sent")
        await asyncio.sleep(0.5)

    async def listen(self):
        """Listen for messages from Gemini Live API"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                logger.debug(f"[GEMINI] Raw message received: {json.dumps(data, indent=2)}")
                
                # Handle setup confirmation
                if "setupComplete" in data:
                    self.setup_id = data.get("setupComplete", {}).get("metadata", {}).get("setupId")
                    logger.info(f"[GEMINI] ✅ Session setup complete: {self.setup_id}")
                
                # Handle server content (audio response)
                elif "serverContent" in data:
                    server_content = data["serverContent"]
                    
                    # Check if it's audio data
                    if "modelTurn" in server_content:
                        model_turn = server_content["modelTurn"]
                        for part in model_turn.get("parts", []):
                            if "inlineData" in part:
                                inline_data = part["inlineData"]
                                # Check for both PCM formats Gemini might send
                                if inline_data.get("mimeType") in ["audio/pcm", "audio/pcm;rate=24000"]:
                                    # Decode base64 audio data
                                    audio_data = base64.b64decode(inline_data["data"])
                                    if audio_data:
                                        timestamp = time.time()
                                        if self.last_audio_time:
                                            delta = timestamp - self.last_audio_time
                                            logger.debug(
                                                f"[GEMINI] Audio chunk: {len(audio_data)} bytes, delta: {delta:.3f}s"
                                            )
                                        else:
                                            logger.info("[GEMINI] Audio stream started")
                                        self.last_audio_time = timestamp

                                        # Send audio to callback
                                        if self.audio_callback:
                                            self.audio_callback(audio_data)

                                        # Send audio levels for visualization
                                        if self.level_callback:
                                            self.level_callback(audio_data)

                                        self.is_speaking = True
                    
                    # ✨ NEW: Capture output transcription (what AI actually said)
                    if "outputTranscription" in server_content:
                        transcription = server_content["outputTranscription"]
                        if "text" in transcription:
                            transcribed_text = transcription["text"]
                            logger.info(f"[GEMINI] 📝 AI speech transcribed: {transcribed_text[:60]}...")
                            self.current_text_response += transcribed_text
                            
                    # ✨ NEW: Capture input transcription (what user said)  
                    if "inputTranscription" in server_content:
                        transcription = server_content["inputTranscription"]
                        if "text" in transcription:
                            transcribed_text = transcription["text"]
                            logger.info(f"[GEMINI] 📝 User speech transcribed: {transcribed_text[:60]}...")
                    
                    # Check if turn is complete
                    if server_content.get("turnComplete"):
                        logger.info("[GEMINI] Audio response completed")
                        self.is_speaking = False
                        self.last_audio_time = None

                        # Send silence to level callback to fade out waveform
                        if self.level_callback:
                            silence = bytes(2048)
                            for _ in range(5):
                                self.level_callback(silence)
                        
                        # ✨ NEW: Send transcribed text to callback if we have one
                        if self.text_response_callback and self.current_text_response.strip():
                            logger.info(f"[GEMINI] 📝 Captured complete response: {self.current_text_response[:100]}...")
                            self.text_response_callback(self.current_text_response.strip())
                            self.current_text_response = ""  # Reset for next response
                        
                        # Notify that response is complete
                        if self.response_complete_callback:
                            self.response_complete_callback()

                # Handle errors
                elif "error" in data:
                    error = data["error"]
                    logger.error(f"[GEMINI] Live API error: {error}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Gemini WebSocket connection closed: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in Gemini WebSocket listener: {e}")
            self.is_connected = False

    async def send_text_message(self, text: str, author: str):
        """Send text message to Gemini Live API for audio response"""
        if not self.is_connected or not self.ws:
            logger.error("Not connected to Gemini Live API")
            return

        try:
            logger.info(f"Sending to Gemini Live API - {author}: {text}")

            # Create client content message with proper format
            client_content = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": text  # Send the text directly, not wrapped with author
                                }
                            ]
                        }
                    ],
                    "turnComplete": True
                }
            }

            await self.ws.send(json.dumps(client_content))
            logger.info("Gemini Live request sent")

        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}")

    async def disconnect(self):
        """Disconnect from Gemini Live API"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("Disconnected from Gemini Live API") 