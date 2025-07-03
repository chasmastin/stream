#!/usr/bin/env python3
"""
Simple Dual Robot Main - Step-by-step approach
Starting with working foundation, adding dual robot features incrementally
"""

import os
import sys
import logging
import argparse

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst

# Initialize GStreamer FIRST
Gst.init(None)

# Import our modules
from modules import config
from modules.streaming.simple_dual_robot_streamer import SimpleDualRobotStreamer

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Simple Dual Robot Streamer - Step by Step"
    )
    parser.add_argument(
        "--stream-key", 
        required=True,
        help="Twitch stream key"
    )

    args = parser.parse_args()

    logger.info("🤖🤖 Starting Simple Dual Robot AI Streamer...")
    logger.info("This version uses the working pipeline foundation with dual robot audio")
    logger.info("No waveforms or complex overlays yet - just basic dual robot conversation")
    
    streamer = SimpleDualRobotStreamer()
    streamer.run(stream_key=args.stream_key)


if __name__ == "__main__":
    main() 