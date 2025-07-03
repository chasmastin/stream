#!/usr/bin/env python3
"""
Simple RTMP Connection Test
Tests if we can connect to Twitch RTMP server with basic video
"""

import os
import sys
import time
import logging

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_rtmp_connection(stream_key: str):
    """Test basic RTMP connection to Twitch"""
    
    rtmp_url = f"rtmp://bog01.contribute.live-video.net/app/{stream_key}"
    logger.info(f"Testing RTMP connection to: {rtmp_url}")
    
    # Super simple pipeline: test pattern -> RTMP
    pipeline_str = f"""
    videotestsrc pattern=0 ! 
    video/x-raw,width=1280,height=720,framerate=30/1 !
    videoconvert ! 
    x264enc bitrate=2500 speed-preset=ultrafast tune=zerolatency !
    h264parse ! 
    flvmux ! 
    rtmpsink location="{rtmp_url}"
    """
    
    logger.info("Creating simple test pipeline...")
    pipeline = Gst.parse_launch(pipeline_str)
    
    if not pipeline:
        logger.error("Failed to create pipeline")
        return False
    
    # Set up message bus for detailed error info
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err}")
            logger.error(f"Debug info: {debug}")
            pipeline.set_state(Gst.State.NULL)
            return False
        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"GStreamer warning: {err}")
        elif message.type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending = message.parse_state_changed()
            if message.src == pipeline:
                logger.info(f"Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")
        elif message.type == Gst.MessageType.STREAM_START:
            logger.info("✅ Stream started successfully!")
        return True
    
    bus.connect("message", on_message)
    
    # Try to start pipeline
    logger.info("Starting test pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("❌ Failed to start pipeline")
        return False
    
    # Wait for state change
    logger.info("Waiting for pipeline state...")
    state_ret, state, pending = pipeline.get_state(5 * Gst.SECOND)  # 5 second timeout
    
    if state_ret == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
        logger.info("✅ Pipeline is PLAYING - RTMP connection successful!")
        logger.info("Streaming test pattern for 10 seconds...")
        time.sleep(10)
        result = True
    else:
        logger.error(f"❌ Pipeline failed to reach PLAYING: {state_ret}, state: {state}")
        result = False
    
    pipeline.set_state(Gst.State.NULL)
    return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RTMP connection")
    parser.add_argument("--stream-key", required=True, help="Twitch stream key")
    
    args = parser.parse_args()
    
    success = test_rtmp_connection(args.stream_key)
    if success:
        logger.info("🎉 RTMP connection test PASSED!")
    else:
        logger.error("💥 RTMP connection test FAILED!")

if __name__ == "__main__":
    main() 