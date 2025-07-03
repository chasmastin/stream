#!/usr/bin/env python3
"""
Dual Robot AI Streamer - Main Entry Point
Two robots having conversations with OpenAI Realtime API
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
from modules.streaming import DualRobotLiveAIStreamer

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable debug logging for dual robot system
logging.getLogger("modules.streaming.dual_robot_streamer").setLevel(logging.DEBUG)
logging.getLogger("modules.api.dual_robot_realtime").setLevel(logging.DEBUG)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual Robot AI Streamer - Two robots having conversations"
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
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("modules.streaming.dual_robot_streamer").setLevel(logging.DEBUG)
        logging.getLogger("modules.api.dual_robot_realtime").setLevel(logging.DEBUG)

    # Create and run dual robot streamer
    streamer = DualRobotLiveAIStreamer()
    
    try:
        logger.info("🤖🤖 Starting Dual Robot AI Streamer...")
        logger.info("Left Robot: Curious and optimistic, loves to start conversations")
        logger.info("Right Robot: Analytical and wise, provides insights and perspectives")
        logger.info("Background video: Two robots will have their waveforms positioned over their faces")
        logger.info("User messages will be stored and incorporated into robot conversations")
        logger.info(f"Platform: {args.platform.upper()}")
        logger.info(f"Stream key: {args.stream_key[:10]}..." if args.stream_key else "No stream key")
        
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
        raise


if __name__ == "__main__":
    main() 