#!/usr/bin/env python3
"""
Basic Dual Robot Main - Step 3
Adding true dual robot API connections with real conversations
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
from modules.api.dual_robot_realtime import DualRobotRealtimeManager

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable debug logging for waveform tracking
logging.getLogger("modules.streaming.streamer").setLevel(logging.DEBUG)


class TrueDualRobotStreamer(WorkingLiveAIStreamer):
    """
    Step 3: Add true dual robot API connections
    - Keep all working video/audio pipeline 
    - Use DualRobotRealtimeManager for real robot conversations
    - Simple audio routing: whichever robot speaks gets sent to stream
    """
    
    def __init__(self):
        super().__init__()
        self.dual_robot_manager = None
        self.current_speaking_robot = None
        logger.info("🤖🤖 TrueDualRobotStreamer initialized")
    
    def start_event_loop(self):
        """Override to use true dual robot connections"""
        logger.info("🤖🤖 Setting up True Dual Robot API connections...")
        
        # Set up callbacks to route robot audio to our existing audio mixer
        def on_left_robot_audio(audio_data):
            """Route left robot audio to the existing audio system"""
            self.current_speaking_robot = 'left'
            logger.debug(f"🔊 Audio from LEFT robot: {len(audio_data)} bytes")
            
            # Feed audio to existing audio mixer (same as single robot)
            if hasattr(self, 'audio_mixer') and self.audio_mixer:
                self.audio_mixer.add_audio_data(audio_data)
        
        def on_right_robot_audio(audio_data):
            """Route right robot audio to the existing audio system"""
            self.current_speaking_robot = 'right'
            logger.debug(f"🔊 Audio from RIGHT robot: {len(audio_data)} bytes")
            
            # Feed audio to existing audio mixer (same as single robot)
            if hasattr(self, 'audio_mixer') and self.audio_mixer:
                self.audio_mixer.add_audio_data(audio_data)
        
        def on_left_robot_level(audio_data):
            """Handle left robot audio levels - use for waveform"""
            if hasattr(self, 'audio_level_callback'):
                self.audio_level_callback(audio_data)
        
        def on_right_robot_level(audio_data):
            """Handle right robot audio levels - use for waveform"""
            if hasattr(self, 'audio_level_callback'):
                self.audio_level_callback(audio_data)
        
        # Create dual robot manager with separate callbacks
        self.dual_robot_manager = DualRobotRealtimeManager(
            left_audio_callback=on_left_robot_audio,
            right_audio_callback=on_right_robot_audio,
            left_level_callback=on_left_robot_level,
            right_level_callback=on_right_robot_level
        )
        
        # Start the dual robot system
        self.dual_robot_manager.start_event_loop()
        
        logger.info("✅ True Dual Robot API connections established!")
        
        # Important: Don't run event loop here since DualRobotRealtimeManager handles it
        # Just set up the manager and let it run in its own thread
    
    def cleanup(self):
        """Clean up dual robot resources"""
        if self.dual_robot_manager:
            logger.info("🛑 Cleaning up dual robot connections...")
            # Add cleanup logic here if DualRobotRealtimeManager has cleanup method
        super().cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="True Dual Robot AI Streamer - Step 3"
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

    logger.info("🤖🤖 Starting True Dual Robot AI Streamer - Step 3...")
    logger.info("📝 This adds REAL dual robot API connections")
    logger.info("🎯 Goal: Two robots having real conversations with each other")
    logger.info("🔊 Audio: Simple routing - whichever robot speaks gets streamed")

    # Use our true dual robot streamer
    streamer = TrueDualRobotStreamer()
    
    try:
        streamer.run(
            platform=args.platform,
            broadcast_id=args.broadcast_id,
            stream_key=args.stream_key,
        )
    except KeyboardInterrupt:
        logger.info("🛑 Dual robot streaming stopped by user")
    except Exception as e:
        logger.error(f"❌ Dual robot streaming failed: {e}")
        import traceback
        traceback.print_exc()
        streamer.cleanup()
        raise


if __name__ == "__main__":
    main() 