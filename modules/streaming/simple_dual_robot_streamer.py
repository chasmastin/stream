"""
Simple Dual Robot Streamer - Step-by-step approach
Starting with working pipeline, adding dual robot audio incrementally
"""

import os
import time
import threading
import logging
import random
from collections import deque

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Import our modules
from .. import config
from ..audio import SimpleAudioMixer, DualAudioMixer
from ..api.dual_robot_realtime import DualRobotRealtimeManager
from ..platforms import TwitchChatSimulator

logger = logging.getLogger(__name__)


class SimpleDualRobotStreamer:
    """Simplified dual robot streamer - step by step approach"""

    def __init__(self):
        self.dual_robot_manager = None
        self.pipeline = None
        self.is_streaming = False
        
        # Use dual audio system for two robots
        self.audio_mixer = DualAudioMixer()
        self.appsrc = None
        
        # Chat monitoring
        self.twitch_chat_sim = None
        
        # Audio timestamps
        self.audio_timestamp = 0
        self.last_audio_time = time.time()
        
        logger.info("[SIMPLE_DUAL] Simple dual robot streamer initialized")

    def left_robot_audio_callback(self, audio_data: bytes):
        """Callback for left robot audio - add to left channel"""
        self.audio_mixer.add_left_audio(audio_data)
        self.last_audio_time = time.time()
        logger.debug(f"[SIMPLE_DUAL] Left robot audio: {len(audio_data)} bytes")

    def right_robot_audio_callback(self, audio_data: bytes):
        """Callback for right robot audio - add to right channel"""
        self.audio_mixer.add_right_audio(audio_data)
        self.last_audio_time = time.time()
        logger.debug(f"[SIMPLE_DUAL] Right robot audio: {len(audio_data)} bytes")

    def left_robot_level_callback(self, audio_data: bytes):
        """Left robot audio levels - placeholder for now"""
        pass

    def right_robot_level_callback(self, audio_data: bytes):
        """Right robot audio levels - placeholder for now"""
        pass

    def check_video_file(self) -> bool:
        """Check if background video file exists"""
        if not os.path.exists(config.BACKGROUND_VIDEO_PATH):
            logger.error(f"Background video not found: {config.BACKGROUND_VIDEO_PATH}")
            return False
        return True

    def setup_gstreamer_pipeline(self, stream_key: str) -> None:
        """Setup simple GStreamer pipeline with exact working audio config"""
        if not self.check_video_file():
            raise FileNotFoundError("Background video file not found")

        # Use Twitch RTMP URL
        rtmp_url = f"{config.TWITCH_RTMP_URL}/{stream_key}"
        logger.info(f"Setting up simple pipeline for TWITCH RTMP: {rtmp_url}")

        # Build pipeline manually like the working version
        logger.info("Creating simple GStreamer pipeline with working audio config...")
        self.pipeline = Gst.Pipeline.new("simple-dual-robot")

        # === VIDEO BRANCH (simplified from working version) ===
        filesrc = Gst.ElementFactory.make("filesrc", "file-source")
        video_path = os.path.abspath(config.BACKGROUND_VIDEO_PATH)
        filesrc.set_property("location", video_path)

        decodebin = Gst.ElementFactory.make("decodebin", "decoder")
        video_queue = Gst.ElementFactory.make("queue", "video-queue")
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
        videoscale = Gst.ElementFactory.make("videoscale", "video-scale")
        
        video_caps = Gst.ElementFactory.make("capsfilter", "video-caps")
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        video_caps.set_property("caps", caps)

        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", config.VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")

        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")

        # === AUDIO BRANCH (exact copy from working version) ===
        self.appsrc = Gst.ElementFactory.make("appsrc", "audio-source")
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", False)  # Changed to False for lower latency
        self.appsrc.set_property("do-timestamp", True)  # FIXED: Changed to True
        self.appsrc.set_property("max-bytes", 100000)
        self.appsrc.set_property("min-latency", 0)  # Added
        self.appsrc.set_property("max-latency", int(0.1 * Gst.SECOND))  # Added

        audio_caps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={config.AUDIO_SAMPLE_RATE},channels={config.AUDIO_CHANNELS},layout=interleaved"
        )
        self.appsrc.set_property("caps", audio_caps)

        audioconvert = Gst.ElementFactory.make("audioconvert", "audio-convert")
        audioresample = Gst.ElementFactory.make("audioresample", "audio-resample")
        audioresample.set_property("quality", 10)  # Maximum quality

        # Audio caps for output
        audio_caps_stereo = Gst.ElementFactory.make("capsfilter", "audio-caps-stereo")
        caps_stereo = Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2")
        audio_caps_stereo.set_property("caps", caps_stereo)

        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", config.AUDIO_BITRATE)

        aacparse = Gst.ElementFactory.make("aacparse", "aac-parser")

        # === MUXER AND OUTPUT ===
        flvmux = Gst.ElementFactory.make("flvmux", "muxer")
        flvmux.set_property("streamable", True)
        flvmux.set_property("latency", 1000000000)

        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")

        rtmpsink.set_property("location", rtmp_url)
        rtmpsink.set_property("async", False)
        rtmpsink.set_property("sync", False)

        # Add all elements to pipeline
        elements = [
            filesrc, decodebin, video_queue, videoconvert, videoscale, video_caps,
            x264enc, h264parse, self.appsrc, audioconvert, audioresample, 
            audio_caps_stereo, audio_encoder, aacparse, flvmux, rtmpsink
        ]

        for element in elements:
            if element:
                self.pipeline.add(element)
            else:
                logger.error(f"Failed to create an element!")

        # Link video chain
        filesrc.link(decodebin)

        # Dynamic pad handling for decoder (simplified)
        def on_pad_added(dbin, pad):
            pad_caps = pad.query_caps(None)
            pad_struct = pad_caps.get_structure(0)
            pad_type = pad_struct.get_name()

            if pad_type.startswith("video/"):
                sink_pad = video_queue.get_static_pad("sink")
                if not sink_pad.is_linked():
                    if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                        logger.info("✅ Video decoder connected")
                    else:
                        logger.error("❌ Failed to link video pad")

        decodebin.connect("pad-added", on_pad_added)

        # Link video chain
        video_queue.link(videoconvert)
        videoconvert.link(videoscale)  
        videoscale.link(video_caps)
        video_caps.link(x264enc)
        x264enc.link(h264parse)

        # Link audio chain (exact copy from working version)
        self.appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audio_caps_stereo)
        audio_caps_stereo.link(audio_encoder)
        audio_encoder.link(aacparse)

        # Link to muxer
        h264parse.link(flvmux)
        aacparse.link(flvmux)
        flvmux.link(rtmpsink)

        # Set up callbacks (simple approach like working version)
        self.appsrc.connect("need-data", self._on_need_data)
        self.appsrc.connect("enough-data", self._on_enough_data)

        # Set up message bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ Simple GStreamer pipeline setup complete")

    def _on_need_data(self, src, length):
        """Audio data callback - simple approach"""
        if not self.is_streaming:
            return
        threading.Thread(target=self._push_audio_data, daemon=True).start()

    def _on_enough_data(self, src):
        """Stop pushing data when enough is buffered"""
        pass

    def _push_audio_data(self):
        """Push mixed audio data to pipeline - simple approach"""
        try:
            chunk_size = int(config.AUDIO_SAMPLE_RATE * 0.02 * 2)  # 20ms chunks
            audio_chunk = self.audio_mixer.get_audio_chunk(chunk_size)
            
            if audio_chunk and len(audio_chunk) > 0:
                buffer = Gst.Buffer.new_allocate(None, len(audio_chunk), None)
                buffer.fill(0, audio_chunk)
                
                # Simple timestamp handling like working version
                buffer.pts = self.audio_timestamp
                buffer.duration = Gst.SECOND * len(audio_chunk) // (config.AUDIO_SAMPLE_RATE * 2)
                self.audio_timestamp += buffer.duration
                
                # Push to pipeline
                ret = self.appsrc.emit("push-buffer", buffer)
                if ret != Gst.FlowReturn.OK:
                    logger.warning(f"[SIMPLE_DUAL] Audio push failed: {ret}")
            else:
                # Push silence if no audio available
                silence = b'\x00' * chunk_size
                buffer = Gst.Buffer.new_allocate(None, len(silence), None)
                buffer.fill(0, silence)
                buffer.pts = self.audio_timestamp
                buffer.duration = Gst.SECOND * len(silence) // (config.AUDIO_SAMPLE_RATE * 2)
                self.audio_timestamp += buffer.duration
                ret = self.appsrc.emit("push-buffer", buffer)

        except Exception as e:
            logger.error(f"[SIMPLE_DUAL] Error pushing audio: {e}")

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages - simple approach"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[SIMPLE_DUAL] GStreamer error: {err}: {debug}")
            self.stop_streaming()
        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"[SIMPLE_DUAL] GStreamer warning: {err}: {debug}")
        elif message.type == Gst.MessageType.EOS:
            logger.info("[SIMPLE_DUAL] End of stream")
            self.stop_streaming()

    def initialize_dual_robot_system(self) -> None:
        """Initialize the dual robot conversation system"""
        logger.info("[SIMPLE_DUAL] Initializing dual robot system...")
        
        self.dual_robot_manager = DualRobotRealtimeManager(
            left_audio_callback=self.left_robot_audio_callback,
            right_audio_callback=self.right_robot_audio_callback,
            left_level_callback=self.left_robot_level_callback,
            right_level_callback=self.right_robot_level_callback
        )
        
        # Start the robot system
        self.dual_robot_manager.start_event_loop()
        
        logger.info("✅ Dual robot system initialized")

    def send_user_message_to_robots(self, message: str, author: str):
        """Send user message to robot conversation system"""
        if self.dual_robot_manager:
            self.dual_robot_manager.add_user_message(author, message)
            logger.info(f"[SIMPLE_DUAL] User message added: {author}: {message}")

    def monitor_twitch_chat(self) -> None:
        """Monitor Twitch chat simulation"""
        logger.info("[SIMPLE_DUAL] Starting Twitch chat simulation...")
        
        while self.is_streaming:
            try:
                message = self.twitch_chat_sim.get_next_message()
                if message:
                    # Send to robots
                    self.send_user_message_to_robots(message['message'], message['author'])
                    logger.info(f"[SIMPLE_DUAL] Chat: {message['author']}: {message['message']}")
                
                time.sleep(random.uniform(5, 15))  # Random intervals like before
                
            except Exception as e:
                logger.error(f"[SIMPLE_DUAL] Twitch chat error: {e}")
                time.sleep(5)

    def start_streaming(self, stream_key: str) -> None:
        """Start the simple dual robot streaming"""
        try:
            self.is_streaming = True
            
            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)
            
            # Initialize dual robot system
            self.initialize_dual_robot_system()
            
            # Setup Twitch chat simulation
            self.twitch_chat_sim = TwitchChatSimulator(config.TWITCH_SIMULATED_MESSAGES)
            threading.Thread(target=self.monitor_twitch_chat, daemon=True).start()
            
            # Start pipeline - simple approach like working version
            logger.info("[SIMPLE_DUAL] Starting GStreamer pipeline...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start GStreamer pipeline")
            
            # Wait for pipeline to reach PLAYING state
            logger.info("[SIMPLE_DUAL] Waiting for pipeline to start...")
            state_ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
            
            if state_ret == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
                logger.info("✅ [SIMPLE_DUAL] Pipeline is now PLAYING")
            else:
                logger.error(f"❌ [SIMPLE_DUAL] Pipeline failed to reach PLAYING state: {state_ret}, current state: {state}")
            
            logger.info("🎬 Simple dual robot streaming started!")
            
            # Keep running
            try:
                while self.is_streaming:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Streaming interrupted by user")
            
        except Exception as e:
            logger.error(f"[SIMPLE_DUAL] Streaming error: {e}")
            raise
        finally:
            self.stop_streaming()

    def stop_streaming(self) -> None:
        """Stop the simple dual robot streaming"""
        logger.info("[SIMPLE_DUAL] Stopping streaming...")
        
        self.is_streaming = False
        
        # Stop dual robot system
        if self.dual_robot_manager:
            self.dual_robot_manager.stop_conversation()
        
        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        logger.info("🛑 Simple dual robot streaming stopped")

    def run(self, stream_key: str) -> None:
        """Run the simple dual robot streamer"""
        logger.info("[SIMPLE_DUAL] Starting simple dual robot streamer for Twitch")
        
        try:
            if not stream_key:
                logger.error("Stream key required for Twitch")
                return
            self.start_streaming(stream_key)
                
        except Exception as e:
            logger.error(f"[SIMPLE_DUAL] Failed to start streaming: {e}")
            raise
