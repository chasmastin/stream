#!/usr/bin/env python3
"""
Dual Robot Simple Main - Final Working Solution
Inherits from WorkingLiveAIStreamer (proven working pipeline)
Adds dual robot capabilities with simple audio routing
"""

import os
import sys
import logging
import argparse
import asyncio
import threading
import time

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst

# Initialize GStreamer FIRST
Gst.init(None)

# Import our modules
from modules import config
from modules.streaming.streamer import WorkingLiveAIStreamer
from modules.api.dual_robot_realtime import DualRobotRealtimeManager

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DualRobotSimpleStreamer(WorkingLiveAIStreamer):
    """
    Dual Robot Streamer that inherits from the working single robot streamer
    Uses the proven working pipeline and adds dual robot conversation
    Simple approach: only one robot audio at a time
    """
    
    def __init__(self):
        super().__init__()
        self.dual_robot_manager = None
        self.current_speaking_robot = None
        
    def initialize_realtime_api(self):
        """Override to use dual robot manager instead of single robot"""
        logger.info("🤖🤖 Setting up Dual Robot Realtime API...")
        
        # Set up callbacks to route audio to our existing audio mixer
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
            """Handle left robot audio levels"""
            pass
        
        def on_right_robot_level(audio_data):
            """Handle right robot audio levels"""
            pass
        
        # Create dual robot manager with separate callbacks
        self.dual_robot_manager = DualRobotRealtimeManager(
            left_audio_callback=on_left_robot_audio,
            right_audio_callback=on_right_robot_audio,
            left_level_callback=on_left_robot_level,
            right_level_callback=on_right_robot_level
        )
        
        # Start the dual robot event loop
        self.dual_robot_manager.start_event_loop()
        
        logger.info("✅ Dual Robot Realtime API setup complete")

    def cleanup(self):
        """Clean up dual robot resources"""
        if self.dual_robot_manager:
            self.dual_robot_manager.cleanup()
        super().cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Dual Robot Simple Streamer - Working Solution"
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
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🤖🤖 Starting Dual Robot Simple AI Streamer...")
    logger.info("💡 This version inherits from the working single robot pipeline")
    logger.info("💡 Adds dual robot conversation with simple audio routing")
    logger.info("💡 One robot audio at a time - no complex mixing")
    
    streamer = DualRobotSimpleStreamer()
    
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
        if args.debug:
            import traceback
            traceback.print_exc()
        streamer.cleanup()
        raise


if __name__ == "__main__":
    main() 