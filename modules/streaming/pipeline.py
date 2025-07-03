"""
GStreamer Pipeline Builder
"""

import os
import logging
from gi.repository import Gst

from .. import config

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Helper class to build GStreamer pipelines"""
    
    @staticmethod
    def check_video_file() -> bool:
        """Check if the video file exists and is valid"""
        if not os.path.exists(config.BACKGROUND_VIDEO_PATH):
            logger.error(f"Video file not found: {config.BACKGROUND_VIDEO_PATH}")
            return False

        # Try to get video info using GStreamer discoverer
        try:
            discoverer = Gst.ElementFactory.make("uridecodebin", None)
            if discoverer:
                logger.info(f"✅ Video file found: {config.BACKGROUND_VIDEO_PATH}")
                return True
        except:
            pass

        return True  # Assume it's valid if it exists

    @staticmethod
    def create_rtmp_location(platform: str, stream_key: str) -> str:
        """Create RTMP URL based on platform"""
        if platform == "youtube":
            return f"{config.YOUTUBE_RTMP_BASE_URL}/{stream_key}"
        else:
            return f"{config.TWITCH_RTMP_URL}/{stream_key}"

    @staticmethod
    def create_video_encoder(platform: str):
        """Create and configure video encoder"""
        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", config.VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("key-int-max", 30 if platform == "youtube" else 60)
        x264enc.set_property("bframes", 0)
        x264enc.set_property("byte-stream", True)
        x264enc.set_property("aud", True)
        x264enc.set_property("cabac", False)
        return x264enc

    @staticmethod
    def create_audio_encoder():
        """Create and configure audio encoder"""
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", config.AUDIO_BITRATE)
        return audio_encoder

    @staticmethod
    def create_rtmp_sink(rtmp_location: str):
        """Create and configure RTMP sink"""
        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")
        
        logger.info(f"📡 RTMP URL: {rtmp_location}")
        rtmpsink.set_property("location", rtmp_location)
        rtmpsink.set_property("async", False)
        rtmpsink.set_property("sync", False)
        return rtmpsink 