#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 2
Adding minimal dual robot support to working main.py structure
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import asyncio
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


class BasicDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Step 2: Add minimal dual robot support
    - Keep all working video/audio pipeline 
    - Just change the AI instructions to mention two robots
    - No complex dual robot API yet
    """
    
    def __init__(self):
        super().__init__()
        logger.info("🤖🤖 BasicDualRobotStreamer initialized")
    
    def start_event_loop(self):
        """Override to use dual robot instructions but same threading as parent"""
        logger.info("🤖🤖 Setting up Basic Dual Robot API (still single connection)...")
        
        # Use the exact same threading setup as parent, but with dual robot instructions
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Create Realtime client with DUAL robot instructions
        dual_robot_instructions = """You are part of a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (Left): Curious, optimistic, loves to start conversations and ask questions
        🤖 Robot-R (Right): Analytical, wise, provides insights and builds on ideas
        
        Sometimes respond as Robot-L, sometimes as Robot-R. Start your messages with [Robot-L] or [Robot-R].
        Keep responses under 20 words. Have natural back-and-forth conversations with yourself!
        
        Example:
        [Robot-L] Hey there! What's the most fascinating thing about AI?
        [Robot-R] The intersection of creativity and logic - we can both dream and calculate simultaneously.
        """
        
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            config.OPENAI_VOICE,
            dual_robot_instructions,  # Use dual robot instructions
            self.audio_received_callback, 
            self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Keep the loop running
        self.event_loop.run_forever()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Basic Dual Robot AI Streamer - Step 2"
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

    logger.info("🤖🤖 Starting Basic Dual Robot AI Streamer - Step 2...")
    logger.info("📝 This adds basic dual robot personality to working pipeline")
    logger.info("🎯 Goal: Same working stream but AI acts like two robots")

    # Use our basic dual robot streamer
    streamer = BasicDualRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 