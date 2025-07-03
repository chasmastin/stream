#!/usr/bin/env python3
"""
Dual Waveform Solution - Two separate waveforms for left/right robots
Instead of moving one waveform, show/hide two separate waveforms
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import asyncio
import random
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


class DualWaveformRobotStreamer(WorkingLiveAIStreamer):
    """
    Two separate waveforms approach
    - Left waveform at (320, 200) 
    - Right waveform at (960, 200)
    - Show only the active robot's waveform
    """
    
    def __init__(self):
        super().__init__()
        self.current_robot_voice = "alloy"
        self.robot_voices = ["alloy", "echo"]
        self.conversation_count = 0
        self.current_speaking_robot = "left"
        self.left_waveform_pad = None
        self.right_waveform_pad = None
        
        logger.info("🤖🌊🌊🤖 DualWaveformRobotStreamer initialized")
        logger.info("🎯 Strategy: Two separate waveforms instead of moving one")
    
    def setup_gstreamer_pipeline(self):
        """Override to create dual waveforms"""
        logger.info("🔧 Setting up dual waveform pipeline...")
        
        # Create pipeline
        self.pipeline = Gst.Pipeline.new("live-streaming-pipeline")
        
        # === VIDEO ELEMENTS ===
        # Background video
        video_source = Gst.ElementFactory.make("filesrc", "video-source")
        video_source.set_property("location", config.BACKGROUND_VIDEO_PATH)
        
        video_decode = Gst.ElementFactory.make("decodebin", "video-decode")
        video_decode.connect("pad-added", self._on_video_pad_added)
        
        # Video processing
        video_convert = Gst.ElementFactory.make("videoconvert", "video-convert")
        video_scale = Gst.ElementFactory.make("videoscale", "video-scale")
        video_rate = Gst.ElementFactory.make("videorate", "video-rate")
        
        video_caps = Gst.Caps.from_string(
            f"video/x-raw,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},"
            f"framerate={config.VIDEO_FRAMERATE}/1,format=I420"
        )
        video_filter = Gst.ElementFactory.make("capsfilter", "video-filter")
        video_filter.set_property("caps", video_caps)
        
        # === DUAL WAVEFORM ELEMENTS ===
        # LEFT waveform
        left_waveform_src = Gst.ElementFactory.make("appsrc", "left-waveform-source")
        left_waveform_src.set_property("caps", 
            Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={config.WAVEFORM_WIDTH},"
                               f"height={config.WAVEFORM_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"))
        left_waveform_src.set_property("format", Gst.Format.TIME)
        left_waveform_src.set_property("is-live", True)
        
        # RIGHT waveform  
        right_waveform_src = Gst.ElementFactory.make("appsrc", "right-waveform-source")
        right_waveform_src.set_property("caps",
            Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={config.WAVEFORM_WIDTH},"
                               f"height={config.WAVEFORM_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"))
        right_waveform_src.set_property("format", Gst.Format.TIME)
        right_waveform_src.set_property("is-live", True)
        
        # === COMPOSITOR ===
        compositor = Gst.ElementFactory.make("compositor", "compositor")
        
        # === AUDIO ELEMENTS ===
        audio_source = Gst.ElementFactory.make("appsrc", "audio-source")
        audio_source.set_property("caps", 
            Gst.Caps.from_string(f"audio/x-raw,format=S16LE,channels={config.AUDIO_CHANNELS},"
                               f"rate={config.AUDIO_SAMPLE_RATE}"))
        audio_source.set_property("format", Gst.Format.TIME)
        audio_source.set_property("is-live", True)
        
        audio_convert = Gst.ElementFactory.make("audioconvert", "audio-convert")
        audio_resample = Gst.ElementFactory.make("audioresample", "audio-resample")
        
        # === ENCODING ===
        video_encoder = Gst.ElementFactory.make("x264enc", "video-encoder")
        video_encoder.set_property("bitrate", config.VIDEO_BITRATE)
        video_encoder.set_property("tune", "zerolatency")
        video_encoder.set_property("speed-preset", "ultrafast")
        
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        audio_encoder.set_property("bitrate", config.AUDIO_BITRATE)
        
        # === MUXER ===
        muxer = Gst.ElementFactory.make("flvmux", "muxer")
        muxer.set_property("streamable", True)
        
        # === SINK ===
        sink = Gst.ElementFactory.make("rtmpsink", "sink")
        
        # Add all elements to pipeline
        elements = [
            video_source, video_decode, video_convert, video_scale, video_rate,
            video_filter, left_waveform_src, right_waveform_src, compositor,
            audio_source, audio_convert, audio_resample,
            video_encoder, audio_encoder, muxer, sink
        ]
        
        for element in elements:
            if not self.pipeline.add(element):
                logger.error(f"❌ Failed to add {element.get_name()} to pipeline")
                return False
        
        # === LINK VIDEO CHAIN ===
        if not video_source.link(video_decode):
            logger.error("❌ Failed to link video source to decode")
            return False
        
        # Video processing chain will be linked in pad-added callback
        
        # === LINK DUAL WAVEFORMS TO COMPOSITOR ===
        # Get compositor sink pads and position them
        video_pad = compositor.get_request_pad("sink_%u")
        left_pad = compositor.get_request_pad("sink_%u") 
        right_pad = compositor.get_request_pad("sink_%u")
        
        # Position left waveform
        left_pad.set_property("xpos", config.ROBOT_POSITIONS['left']['x'] - config.WAVEFORM_WIDTH // 2)
        left_pad.set_property("ypos", config.ROBOT_POSITIONS['left']['y'])
        left_pad.set_property("zorder", 1)
        left_pad.set_property("alpha", 1.0)  # Start visible
        
        # Position right waveform  
        right_pad.set_property("xpos", config.ROBOT_POSITIONS['right']['x'] - config.WAVEFORM_WIDTH // 2)
        right_pad.set_property("ypos", config.ROBOT_POSITIONS['right']['y'])
        right_pad.set_property("zorder", 1)
        right_pad.set_property("alpha", 0.0)  # Start hidden
        
        # Store pad references
        self.left_waveform_pad = left_pad
        self.right_waveform_pad = right_pad
        
        # Link waveforms to compositor
        left_waveform_src.get_static_pad("src").link(left_pad)
        right_waveform_src.get_static_pad("src").link(right_pad)
        
        # === LINK AUDIO CHAIN ===
        if not audio_source.link_many(audio_convert, audio_resample, audio_encoder):
            logger.error("❌ Failed to link audio chain")
            return False
        
        # === LINK FINAL ENCODING ===
        if not compositor.link_many(video_encoder, muxer):
            logger.error("❌ Failed to link video encoding")
            return False
        
        if not audio_encoder.link(muxer):
            logger.error("❌ Failed to link audio to muxer")
            return False
        
        if not muxer.link(sink):
            logger.error("❌ Failed to link muxer to sink")
            return False
        
        # Store source references
        self.audio_source = audio_source
        self.left_waveform_source = left_waveform_src
        self.right_waveform_source = right_waveform_src
        self.rtmp_sink = sink
        
        logger.info("✅ Dual waveform pipeline setup complete")
        return True
    
    def _on_video_pad_added(self, decodebin, pad):
        """Handle dynamic video pad"""
        logger.info(f"🔗 Video pad added: {pad.get_name()}")
        
        video_convert = self.pipeline.get_by_name("video-convert")
        if video_convert:
            sink_pad = video_convert.get_static_pad("sink")
            if not sink_pad.is_linked():
                if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                    logger.info("✅ Video pad linked successfully")
                    
                    # Link rest of video chain
                    video_convert = self.pipeline.get_by_name("video-convert")
                    video_scale = self.pipeline.get_by_name("video-scale")
                    video_rate = self.pipeline.get_by_name("video-rate")
                    video_filter = self.pipeline.get_by_name("video-filter")
                    compositor = self.pipeline.get_by_name("compositor")
                    
                    video_convert.link_many(video_scale, video_rate, video_filter)
                    
                    # Link to compositor background pad
                    video_pad = compositor.get_request_pad("sink_%u")
                    video_pad.set_property("zorder", 0)  # Background layer
                    video_filter.get_static_pad("src").link(video_pad)
                    
                else:
                    logger.error("❌ Failed to link video pad")
    
    def start_event_loop(self):
        """Keep Step 3a event loop with dual waveform control"""
        logger.info("🤖🌊🌊🤖 Setting up Dual Waveform Robot API...")
        
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        dual_robot_instructions = """You are a dual robot AI system! You can roleplay as BOTH robots:
        
        🤖 Robot-L (LEFT): Curious, optimistic, positioned on the LEFT side
        🤖 Robot-R (RIGHT): Analytical, wise, positioned on the RIGHT side
        
        Always start messages with [Robot-L] or [Robot-R]. Keep responses under 20 words.
        Reference your left/right positions occasionally.
        """
        
        self.current_robot_voice = random.choice(self.robot_voices)
        
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            self.current_robot_voice,
            dual_robot_instructions,
            self.audio_received_callback,
            self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Schedule conversations
        self.event_loop.call_later(10.0, self._start_robot_conversation)

        self.event_loop.run_forever()
    
    def send_message_to_realtime(self, message: str, author: str):
        """Track robot and switch waveforms"""
        super().send_message_to_realtime(message, author)
        
        if "[Robot-L]" in message:
            self.current_speaking_robot = "left"
            logger.info("🤖⬅️ LEFT robot speaking - showing LEFT waveform")
            self._switch_to_left_waveform()
        elif "[Robot-R]" in message:
            self.current_speaking_robot = "right"
            logger.info("🤖➡️ RIGHT robot speaking - showing RIGHT waveform")
            self._switch_to_right_waveform()
    
    def _switch_to_left_waveform(self):
        """Show left waveform, hide right"""
        GLib.idle_add(self._do_switch_left)
    
    def _switch_to_right_waveform(self):
        """Show right waveform, hide left"""
        GLib.idle_add(self._do_switch_right)
    
    def _do_switch_left(self):
        """Actually switch to left waveform"""
        try:
            if self.left_waveform_pad and self.right_waveform_pad:
                self.left_waveform_pad.set_property("alpha", 1.0)  # Show left
                self.right_waveform_pad.set_property("alpha", 0.0)  # Hide right
                logger.info("✅ Switched to LEFT waveform")
        except Exception as e:
            logger.error(f"❌ Failed to switch to left waveform: {e}")
        return False
    
    def _do_switch_right(self):
        """Actually switch to right waveform"""
        try:
            if self.left_waveform_pad and self.right_waveform_pad:
                self.left_waveform_pad.set_property("alpha", 0.0)  # Hide left
                self.right_waveform_pad.set_property("alpha", 1.0)  # Show right
                logger.info("✅ Switched to RIGHT waveform")
        except Exception as e:
            logger.error(f"❌ Failed to switch to right waveform: {e}")
        return False
    
    def _start_robot_conversation(self):
        """Start robot conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            if self.conversation_count % 2 == 0:
                starter = "[Robot-L] Hello from the LEFT side!"
                self.current_speaking_robot = "left"
            else:
                starter = "[Robot-R] Greetings from the RIGHT side!"
                self.current_speaking_robot = "right"
            
            asyncio.create_task(self.realtime_client.send_text_message(starter, "System"))
            self.conversation_count += 1
            
            self.event_loop.call_later(15.0, self._continue_robot_conversation)
    
    def _continue_robot_conversation(self):
        """Continue conversations"""
        if self.realtime_client and self.realtime_client.is_connected:
            is_left_turn = (self.conversation_count % 2 == 0)
            
            if is_left_turn:
                topics = ["[Robot-L] Robot-R, what's your RIGHT-side perspective?"]
                self.current_speaking_robot = "left"
            else:
                topics = ["[Robot-R] Robot-L, from the LEFT, what do you think?"]
                self.current_speaking_robot = "right"
            
            topic = random.choice(topics)
            asyncio.create_task(self.realtime_client.send_text_message(topic, "RobotConversation"))
            self.conversation_count += 1
            
            # Switch voice
            current_index = self.robot_voices.index(self.current_robot_voice)
            next_index = (current_index + 1) % len(self.robot_voices)
            self.current_robot_voice = self.robot_voices[next_index]
            
            next_conversation = random.uniform(12.0, 18.0)
            self.event_loop.call_later(next_conversation, self._continue_robot_conversation)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dual Waveform Robot Streamer")
    parser.add_argument("--platform", choices=["youtube", "twitch"], required=True)
    parser.add_argument("--broadcast-id", help="YouTube broadcast ID")
    parser.add_argument("--stream-key", help="Stream key")

    args = parser.parse_args()

    logger.info("🤖🌊🌊🤖 Starting Dual Waveform Robot Streamer...")
    logger.info("🎯 Strategy: Two separate waveforms at fixed positions")

    streamer = DualWaveformRobotStreamer()
    streamer.run(
        platform=args.platform,
        broadcast_id=args.broadcast_id,
        stream_key=args.stream_key,
    )


if __name__ == "__main__":
    main() 