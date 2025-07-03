"""
Main Live AI Streamer class
"""

import os
import sys
import time
import threading
import logging
import random
import asyncio
import numpy as np
from collections import deque
from typing import Optional

# GStreamer imports
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib

# Import our modules
from .. import config
from ..audio import AudioWaveformGenerator, SimpleAudioMixer
from ..api import SimpleRealtimeAPIClient
from ..platforms import YouTubeAPIManager, TwitchChatSimulator

logger = logging.getLogger(__name__)


class WorkingLiveAIStreamer:
    """Working multi-platform streamer with Ultra-Fast chat monitoring and audio waveform"""

    def __init__(self):
        self.platform = None
        self.youtube_api = None
        self.twitch_chat_sim = None
        self.realtime_client = None
        self.pipeline = None
        self.is_streaming = False
        self.audio_mixer = SimpleAudioMixer()
        self.appsrc = None
        self.text_overlay = None
        self.waveform_overlay = None
        self.waveform_generator = AudioWaveformGenerator(
            width=config.WAVEFORM_WIDTH,
            height=config.WAVEFORM_HEIGHT,
            bars=config.WAVEFORM_BARS
        )
        print(f"[WAVEFORM DEBUG] Generator initialized: {config.WAVEFORM_WIDTH}x{config.WAVEFORM_HEIGHT} with {config.WAVEFORM_BARS} bars")
        self.waveform_appsrc = None
        # Use deque for ultra-fast message processing
        self.processed_messages = deque(maxlen=500)
        self.message_hashes = set()
        self.audio_timestamp = 0
        self.waveform_timestamp = 0
        self.event_loop = None
        self.loop_thread = None
        self.last_audio_time = time.time()
        self.buffer_monitor_thread = None
        self.waveform_levels = np.array([0.0])  # Single value for mouth openness
        self.waveform_lock = threading.Lock()
        self.waveform_update_thread = None
        self.waveform_running = False
        # Add variables for mouth-like animation
        self.waveform_animation_speed = 0.20  # Faster speed for mouth opening/closing
        self.waveform_mouth_openness = 0.0  # How open the mouth is (0-1)
        self.waveform_target_openness = 0.0  # Target openness

    def audio_received_callback(self, audio_data: bytes):
        """Callback for audio from Realtime API"""
        self.audio_mixer.add_audio(audio_data)
        self.last_audio_time = time.time()

    def audio_level_callback(self, audio_data: bytes):
        """Callback for audio levels - simplified since waveform uses random animation"""
        # Since we're using random animation instead of audio levels,
        # we just need this callback to exist for the API but don't need to process the levels
        pass

    def start_event_loop(self):
        """Start asyncio event loop for Realtime API"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Create and connect Realtime client
        self.realtime_client = SimpleRealtimeAPIClient(
            config.OPENAI_API_KEY, 
            config.OPENAI_REALTIME_URL,
            config.OPENAI_REALTIME_MODEL,
            config.OPENAI_VOICE,
            config.AI_INSTRUCTIONS,
            self.audio_received_callback, 
            self.audio_level_callback
        )
        self.event_loop.run_until_complete(self.realtime_client.connect())

        # Keep the loop running
        self.event_loop.run_forever()

    def monitor_audio_buffer(self):
        """Monitor audio buffer health with quality-focused approach"""
        last_status = None
        critical_count = 0
        last_buffer_size = 0
        burst_active = False

        while self.is_streaming:
            try:
                health = self.audio_mixer.get_buffer_health()
                status = health["status"]
                buffer_seconds = health["seconds"]
                
                # Get incoming rate
                incoming_rate = getattr(self.audio_mixer, 'last_incoming_rate', 0.0)
                
                # Detect burst conditions
                if incoming_rate > 3.0 and not burst_active:
                    burst_active = True
                    logger.info(f"Buffer monitor: BURST DETECTED ({incoming_rate:.1f}x incoming)")
                elif incoming_rate < 1.5 and burst_active:
                    burst_active = False
                    logger.info(f"Buffer monitor: Burst ended, normalizing")

                # Track if buffer is meaningfully shrinking
                is_shrinking = buffer_seconds < last_buffer_size - 0.2  # Need 0.2s shrinkage
                last_buffer_size = buffer_seconds

                # Only log when status changes or when very high
                if status != last_status or buffer_seconds > 15.0:
                    if status == "empty":
                        # This is normal when not speaking
                        if health["is_speaking"]:
                            logger.debug(f"Buffer empty during speech")
                    elif status == "healthy":
                        logger.debug(f"Buffer healthy: {buffer_seconds:.2f}s")
                        critical_count = 0  # Reset counter
                    elif status == "filling":
                        logger.info(
                            f"Buffer filling: {buffer_seconds:.2f}s ({health['chunks']} chunks)"
                        )
                    elif status == "warning":
                        if burst_active:
                            logger.info(
                                f"Buffer growing during burst: {buffer_seconds:.2f}s (incoming: {incoming_rate:.1f}x) - quality-focused handling"
                            )
                        else:
                            logger.info(
                                f"Buffer growing: {buffer_seconds:.2f}s - gentle speed adjustment for quality"
                            )
                    elif status == "critical":
                        critical_count += 1
                        
                        if burst_active:
                            if buffer_seconds > 22.0:
                                logger.warning(
                                    f"Buffer very large during burst: {buffer_seconds:.2f}s (incoming: {incoming_rate:.1f}x) - approaching emergency limit"
                                )
                            else:
                                logger.info(
                                    f"Buffer elevated during burst: {buffer_seconds:.2f}s (incoming: {incoming_rate:.1f}x) - quality-preserving catch-up"
                                )
                        elif is_shrinking:
                            logger.info(
                                f"Buffer high but shrinking: {buffer_seconds:.2f}s (quality speed active)"
                            )
                        elif buffer_seconds > 22.0:
                            logger.warning(
                                f"Buffer very high: {buffer_seconds:.2f}s - emergency clear approaching at 25s"
                            )
                        elif buffer_seconds > 15.0:
                            logger.warning(
                                f"Buffer high: {buffer_seconds:.2f}s - moderate quality-preserving catch-up"
                            )

                    last_status = status

                # Reset critical count if status improves or buffer is shrinking
                if status != "critical" or is_shrinking:
                    if critical_count > 0:
                        logger.info("Buffer recovering, resetting critical counter")
                    critical_count = 0

                # Dynamic sleep based on buffer status and burst mode
                if burst_active:
                    time.sleep(0.1)  # Check frequently during bursts
                elif buffer_seconds > 15.0:
                    time.sleep(0.2)  # Check frequently when very high
                elif buffer_seconds > 8.0:
                    time.sleep(0.3)  # Check regularly when elevated
                elif status == "critical":
                    time.sleep(0.5)  # Check less frequently
                elif status == "warning":
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)  # Check infrequently when healthy

            except Exception as e:
                logger.error(f"Buffer monitor error: {e}")
                time.sleep(2)

    def check_video_file(self) -> bool:
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

    def setup_gstreamer_pipeline(self, stream_key: str) -> None:
        """Create working GStreamer pipeline with video file and waveform overlay"""
        logger.info("Creating GStreamer pipeline with video background and waveform...")

        # Check video file first
        if not self.check_video_file():
            logger.warning("Video file not found, falling back to test pattern")
            self.setup_gstreamer_pipeline_fallback(stream_key)
            return

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("ai-streamer")

        # === VIDEO BRANCH ===
        # File source
        filesrc = Gst.ElementFactory.make("filesrc", "file-source")
        video_path = os.path.abspath(config.BACKGROUND_VIDEO_PATH)
        logger.info(f"Setting video source to: {video_path}")
        filesrc.set_property("location", video_path)

        # Queue after file source
        file_queue = Gst.ElementFactory.make("queue", "file-queue")
        file_queue.set_property("max-size-buffers", 200)
        file_queue.set_property("max-size-time", 0)
        file_queue.set_property("max-size-bytes", 0)

        # Decode bin
        decodebin = Gst.ElementFactory.make("decodebin", "decoder")

        # Video queue for buffering
        video_queue = Gst.ElementFactory.make("queue", "video-queue")
        video_queue.set_property("max-size-time", 2 * Gst.SECOND)
        video_queue.set_property("max-size-bytes", 0)
        video_queue.set_property("max-size-buffers", 0)

        # Video convert
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")

        # Video scale - with proper scaling method
        videoscale = Gst.ElementFactory.make("videoscale", "video-scale")
        videoscale.set_property("method", 1)  # Bilinear scaling
        videoscale.set_property("add-borders", False)  # No letterboxing

        # Video rate
        videorate = Gst.ElementFactory.make("videorate", "video-rate")

        # Video caps - FULL WIDTH AND HEIGHT with aspect ratio
        video_caps = Gst.ElementFactory.make("capsfilter", "video-caps")
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1,pixel-aspect-ratio=1/1"
        )
        video_caps.set_property("caps", caps)

        # === WAVEFORM OVERLAY ===
        # Waveform video source (appsrc for dynamic waveform)
        self.waveform_appsrc = Gst.ElementFactory.make("appsrc", "waveform-source")
        self.waveform_appsrc.set_property("format", Gst.Format.TIME)
        self.waveform_appsrc.set_property("is-live", True)
        self.waveform_appsrc.set_property("block", False)
        self.waveform_appsrc.set_property("do-timestamp", True)

        waveform_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={config.WAVEFORM_WIDTH},height={config.WAVEFORM_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        self.waveform_appsrc.set_property("caps", waveform_caps)

        # Waveform video convert
        waveform_convert = Gst.ElementFactory.make("videoconvert", "waveform-convert")

        # Video mixer to overlay waveform
        compositor = Gst.ElementFactory.make("compositor", "compositor")
        # IMPORTANT: Set compositor to transparent background (not black)
        compositor.set_property("background", 0)  # 0 = transparent

        # Create a caps filter after compositor to force output size
        compositor_caps = Gst.ElementFactory.make("capsfilter", "compositor-caps")
        compositor_output_caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        compositor_caps.set_property("caps", compositor_output_caps)

        # Convert back after compositor
        post_compositor_convert = Gst.ElementFactory.make(
            "videoconvert", "post-compositor-convert"
        )

        # Text overlay
        self.text_overlay = Gst.ElementFactory.make("textoverlay", "text-overlay")
        platform_text = "YouTube" if self.platform == "youtube" else "Twitch"
        self.text_overlay.set_property("text", f"HumanHeart.AI")
        self.text_overlay.set_property("valignment", "bottom")
        self.text_overlay.set_property("halignment", "center")
        self.text_overlay.set_property("font-desc", "Sans Bold 18")
        self.text_overlay.set_property("shaded-background", True)
        self.text_overlay.set_property("ypad", 20)  # Add padding from bottom

        # Video encoder with specific settings for live streaming
        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", config.VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("key-int-max", 30)
        x264enc.set_property("bframes", 0)
        x264enc.set_property("byte-stream", True)
        x264enc.set_property("aud", True)
        x264enc.set_property("cabac", False)

        # Video parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        h264parse.set_property("config-interval", 1)

        # === QUEUES FOR MUXER ===
        # Add queues before muxer to handle latency issues
        video_mux_queue = Gst.ElementFactory.make("queue", "video-mux-queue")
        video_mux_queue.set_property("max-size-time", 0)
        video_mux_queue.set_property("max-size-bytes", 0)
        video_mux_queue.set_property("max-size-buffers", 0)

        audio_mux_queue = Gst.ElementFactory.make("queue", "audio-mux-queue")
        audio_mux_queue.set_property("max-size-time", 0)
        audio_mux_queue.set_property("max-size-bytes", 0)
        audio_mux_queue.set_property("max-size-buffers", 0)

        # === AUDIO BRANCH ===
        # Audio source - FIXED TIMESTAMPING
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

        # Audio convert
        audioconvert = Gst.ElementFactory.make("audioconvert", "audio-convert")

        # Audio filter for noise reduction
        audiofilter = Gst.ElementFactory.make("audiodynamic", "audio-filter")
        if audiofilter:
            audiofilter.set_property("mode", 0)  # Compressor mode
            audiofilter.set_property("ratio", 0.5)
            audiofilter.set_property("threshold", 0.3)

        # Audio resample with better quality to reduce noise
        audioresample = Gst.ElementFactory.make("audioresample", "audio-resample")
        audioresample.set_property("quality", 10)  # Maximum quality

        # Audio caps for output
        audio_caps_stereo = Gst.ElementFactory.make("capsfilter", "audio-caps-stereo")
        caps_stereo = Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2")
        audio_caps_stereo.set_property("caps", caps_stereo)

        # Audio encoder
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", config.AUDIO_BITRATE)

        # AAC parse
        aacparse = Gst.ElementFactory.make("aacparse", "aac-parser")

        # === MUXER AND OUTPUT ===
        # FLV muxer
        flvmux = Gst.ElementFactory.make("flvmux", "muxer")
        flvmux.set_property("streamable", True)
        flvmux.set_property("latency", 1000000000)

        # Output queue
        output_queue = Gst.ElementFactory.make("queue2", "output-queue")
        output_queue.set_property("max-size-time", 0)
        output_queue.set_property("max-size-bytes", 1024 * 1024)
        output_queue.set_property("max-size-buffers", 0)
        output_queue.set_property("use-buffering", False)

        # RTMP sink
        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")

        if self.platform == "youtube":
            rtmp_location = f"{config.YOUTUBE_RTMP_BASE_URL}/{stream_key}"
        else:
            # For Twitch, don't add "live=1" parameter
            rtmp_location = f"{config.TWITCH_RTMP_URL}/{stream_key}"

        logger.info(f"📡 RTMP URL: {rtmp_location}")
        rtmpsink.set_property("location", rtmp_location)
        rtmpsink.set_property("async", False)
        rtmpsink.set_property("sync", False)

        # Add all elements to pipeline
        elements = [
            filesrc,
            file_queue,
            decodebin,
            video_queue,
            videoconvert,
            videoscale,
            videorate,
            video_caps,
            compositor,
            compositor_caps,
            post_compositor_convert,
            self.text_overlay,
            x264enc,
            h264parse,
            video_mux_queue,
            self.waveform_appsrc,
            waveform_convert,
            self.appsrc,
            audioconvert,
            audioresample,
            audio_caps_stereo,
            audio_encoder,
            aacparse,
            audio_mux_queue,
            flvmux,
            output_queue,
            rtmpsink,
        ]

        if audiofilter:
            elements.insert(elements.index(audioresample), audiofilter)

        for element in elements:
            if element:
                self.pipeline.add(element)
            else:
                logger.error(f"Failed to create an element!")

        # Link file source chain
        filesrc.link(file_queue)
        file_queue.link(decodebin)

        # Dynamic pad handling for decoder
        def on_pad_added(dbin, pad):
            pad_caps = pad.query_caps(None)
            pad_struct = pad_caps.get_structure(0)
            pad_type = pad_struct.get_name()

            logger.info(f"Decoder pad added: {pad_type}")

            if pad_type.startswith("video/"):
                sink_pad = video_queue.get_static_pad("sink")
                if not sink_pad.is_linked():
                    if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                        logger.info("✅ Video decoder connected to pipeline")
                    else:
                        logger.error("❌ Failed to link video pad")
            elif pad_type.startswith("audio/"):
                logger.info("Ignoring audio track from video file")

        decodebin.connect("pad-added", on_pad_added)

        # Link video chain
        video_queue.link(videoconvert)
        videoconvert.link(videoscale)
        videoscale.link(videorate)
        videorate.link(video_caps)

        # Connect video to compositor background pad
        video_pad = compositor.get_static_pad("sink_0")
        if not video_pad:
            video_pad = compositor.request_pad_simple("sink_%u")

        # CRITICAL: Set video pad properties to fill entire output
        video_pad.set_property("width", config.VIDEO_WIDTH)
        video_pad.set_property("height", config.VIDEO_HEIGHT)
        video_pad.set_property("xpos", 0)
        video_pad.set_property("ypos", 0)
        video_pad.set_property("alpha", 1.0)  # Fully opaque
        video_pad.set_property("zorder", 0)  # Background layer

        # Also set the sizing-policy to keep aspect ratio but fill frame
        try:
            video_pad.set_property("sizing-policy", 1)  # 1 = KEEP_ASPECT_RATIO_OR_SCALE
        except:
            # Some versions might not have this property
            pass

        video_caps.get_static_pad("src").link(video_pad)

        # Connect waveform to compositor overlay pad
        self.waveform_appsrc.link(waveform_convert)
        waveform_pad = compositor.get_static_pad("sink_1")
        if not waveform_pad:
            waveform_pad = compositor.request_pad_simple("sink_%u")

        waveform_x = (config.VIDEO_WIDTH - config.WAVEFORM_WIDTH) // 2  # Center horizontally
        waveform_y = config.VIDEO_HEIGHT - config.WAVEFORM_HEIGHT - 220

        # Position waveform in center
        waveform_pad.set_property("xpos", waveform_x)
        waveform_pad.set_property("ypos", waveform_y)
        waveform_pad.set_property("alpha", 0.8)  # Slightly transparent
        waveform_pad.set_property("zorder", 1)  # Ensure it's on top

        waveform_convert.get_static_pad("src").link(waveform_pad)

        # Link compositor to caps filter then to rest of pipeline
        compositor.link(compositor_caps)
        compositor_caps.link(post_compositor_convert)
        post_compositor_convert.link(self.text_overlay)
        self.text_overlay.link(x264enc)
        x264enc.link(h264parse)
        h264parse.link(video_mux_queue)

        # Link audio chain
        self.appsrc.link(audioconvert)
        audioconvert.link(audiofilter if audiofilter else audioresample)
        if audiofilter:
            audiofilter.link(audioresample)
        audioresample.link(audio_caps_stereo)
        audio_caps_stereo.link(audio_encoder)
        audio_encoder.link(aacparse)
        aacparse.link(audio_mux_queue)

        # Get pads and link to muxer
        try:
            video_pad = flvmux.request_pad_simple("video")
            audio_pad = flvmux.request_pad_simple("audio")
        except AttributeError:
            video_pad = flvmux.get_request_pad("video")
            audio_pad = flvmux.get_request_pad("audio")

        video_mux_queue.get_static_pad("src").link(video_pad)
        audio_mux_queue.get_static_pad("src").link(audio_pad)

        # Link muxer to output
        flvmux.link(output_queue)
        output_queue.link(rtmpsink)

        # Connect signals
        self.appsrc.connect("need-data", self._on_need_data)
        self.appsrc.connect("enough-data", self._on_enough_data)

        self.waveform_appsrc.connect("need-data", self._on_waveform_need_data)

        # Initialize audio and waveform push
        GLib.timeout_add(1, self._push_audio_data)  # Check every 1ms for precise timing
        GLib.timeout_add(33, self._push_waveform_data)  # ~30fps for waveform

        # Bus setup
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ Pipeline configured with video background and waveform overlay")

    def setup_gstreamer_pipeline_fallback(self, stream_key: str) -> None:
        """Fallback pipeline with test pattern if video file is not available"""
        logger.info(
            "Creating fallback GStreamer pipeline with test pattern and waveform..."
        )

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("ai-streamer")

        # Video source - test pattern
        videosrc = Gst.ElementFactory.make("videotestsrc", "video-source")
        videosrc.set_property("pattern", "smpte")
        videosrc.set_property("is-live", True)

        # Video caps
        video_caps = Gst.ElementFactory.make("capsfilter", "video-caps")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        video_caps.set_property("caps", caps)

        # Video convert
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")

        # === WAVEFORM OVERLAY ===
        # Waveform video source
        self.waveform_appsrc = Gst.ElementFactory.make("appsrc", "waveform-source")
        self.waveform_appsrc.set_property("format", Gst.Format.TIME)
        self.waveform_appsrc.set_property("is-live", True)
        self.waveform_appsrc.set_property("block", False)
        self.waveform_appsrc.set_property("do-timestamp", True)

        waveform_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={config.WAVEFORM_WIDTH},height={config.WAVEFORM_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        self.waveform_appsrc.set_property("caps", waveform_caps)

        # Waveform video convert
        waveform_convert = Gst.ElementFactory.make("videoconvert", "waveform-convert")

        # Video mixer
        compositor = Gst.ElementFactory.make("compositor", "compositor")
        compositor.set_property("background", 0)  # Transparent background

        # Compositor caps filter
        compositor_caps = Gst.ElementFactory.make("capsfilter", "compositor-caps")
        compositor_output_caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1"
        )
        compositor_caps.set_property("caps", compositor_output_caps)

        # Convert back after compositor
        post_compositor_convert = Gst.ElementFactory.make(
            "videoconvert", "post-compositor-convert"
        )

        # Text overlay
        self.text_overlay = Gst.ElementFactory.make("textoverlay", "text-overlay")
        platform_text = "YouTube" if self.platform == "youtube" else "Twitch"
        self.text_overlay.set_property(
            "text", f"🤖 {platform_text} AI Stream - Ready! (No Video)"
        )
        self.text_overlay.set_property("valignment", "bottom")
        self.text_overlay.set_property("halignment", "center")
        self.text_overlay.set_property("font-desc", "Sans Bold 16")
        self.text_overlay.set_property("shaded-background", True)

        # Video encoder
        x264enc = Gst.ElementFactory.make("x264enc", "video-encoder")
        x264enc.set_property("bitrate", config.VIDEO_BITRATE)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("key-int-max", 60)
        x264enc.set_property("bframes", 0)

        # Video parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")

        # Audio source - FIXED TIMESTAMPING
        self.appsrc = Gst.ElementFactory.make("appsrc", "audio-source")
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", False)
        self.appsrc.set_property("min-latency", 0)
        self.appsrc.set_property("max-latency", int(0.1 * Gst.SECOND))
        self.appsrc.set_property("do-timestamp", True)  # Already True - good

        audio_caps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={config.AUDIO_SAMPLE_RATE},channels={config.AUDIO_CHANNELS},layout=interleaved"
        )
        self.appsrc.set_property("caps", audio_caps)

        # Audio processing
        audioconvert = Gst.ElementFactory.make("audioconvert", "audio-convert")
        audioresample = Gst.ElementFactory.make("audioresample", "audio-resample")
        audioresample.set_property("quality", 10)  # Maximum quality

        # Convert to stereo
        audio_caps_stereo = Gst.ElementFactory.make("capsfilter", "audio-caps-stereo")
        caps_stereo = Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2")
        audio_caps_stereo.set_property("caps", caps_stereo)

        # Audio encoder
        audio_encoder = Gst.ElementFactory.make("voaacenc", "audio-encoder")
        if not audio_encoder:
            audio_encoder = Gst.ElementFactory.make("avenc_aac", "audio-encoder")
        audio_encoder.set_property("bitrate", config.AUDIO_BITRATE)

        # AAC parser
        aacparse = Gst.ElementFactory.make("aacparse", "aac-parser")

        # === QUEUES FOR MUXER ===
        # Add queues before muxer to handle latency
        video_mux_queue = Gst.ElementFactory.make("queue", "video-mux-queue")
        video_mux_queue.set_property("max-size-time", 0)
        video_mux_queue.set_property("max-size-bytes", 0)
        video_mux_queue.set_property("max-size-buffers", 0)

        audio_mux_queue = Gst.ElementFactory.make("queue", "audio-mux-queue")
        audio_mux_queue.set_property("max-size-time", 0)
        audio_mux_queue.set_property("max-size-bytes", 0)
        audio_mux_queue.set_property("max-size-buffers", 0)

        # Muxer
        flvmux = Gst.ElementFactory.make("flvmux", "muxer")
        flvmux.set_property("streamable", True)
        flvmux.set_property("latency", 40000000)  # 40ms

        # Queue before RTMP
        queue = Gst.ElementFactory.make("queue", "queue")
        queue.set_property("max-size-time", 0)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-buffers", 0)

        # RTMP sink
        rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        if not rtmpsink:
            rtmpsink = Gst.ElementFactory.make("rtmp2sink", "rtmp-sink")

        if self.platform == "youtube":
            rtmp_location = f"{config.YOUTUBE_RTMP_BASE_URL}/{stream_key}"
        else:
            rtmp_location = f"{config.TWITCH_RTMP_URL}/{stream_key}"

        logger.info(f"📡 RTMP URL: {rtmp_location}")
        rtmpsink.set_property("location", rtmp_location)
        rtmpsink.set_property("sync", False)

        # Add all elements
        elements = [
            videosrc,
            video_caps,
            videoconvert,
            compositor,
            compositor_caps,
            post_compositor_convert,
            self.text_overlay,
            x264enc,
            h264parse,
            video_mux_queue,
            self.waveform_appsrc,
            waveform_convert,
            self.appsrc,
            audioconvert,
            audioresample,
            audio_caps_stereo,
            audio_encoder,
            aacparse,
            audio_mux_queue,
            flvmux,
            queue,
            rtmpsink,
        ]

        for element in elements:
            self.pipeline.add(element)

        # Link video chain to compositor
        videosrc.link(video_caps)
        video_caps.link(videoconvert)

        # Connect to compositor
        video_pad = compositor.get_static_pad("sink_0")
        if not video_pad:
            video_pad = compositor.request_pad_simple("sink_%u")

        # Set video pad to fill entire frame
        video_pad.set_property("width", config.VIDEO_WIDTH)
        video_pad.set_property("height", config.VIDEO_HEIGHT)
        video_pad.set_property("xpos", 0)
        video_pad.set_property("ypos", 0)
        video_pad.set_property("alpha", 1.0)
        video_pad.set_property("zorder", 0)

        videoconvert.get_static_pad("src").link(video_pad)

        # Connect waveform
        self.waveform_appsrc.link(waveform_convert)
        waveform_pad = compositor.get_static_pad("sink_1")
        if not waveform_pad:
            waveform_pad = compositor.request_pad_simple("sink_%u")

        # Position waveform in center
        waveform_pad.set_property("xpos", (config.VIDEO_WIDTH - config.WAVEFORM_WIDTH) // 2)
        waveform_pad.set_property("ypos", (config.VIDEO_HEIGHT - config.WAVEFORM_HEIGHT) // 2)
        waveform_pad.set_property("alpha", 0.8)
        waveform_pad.set_property("zorder", 1)

        waveform_convert.get_static_pad("src").link(waveform_pad)

        # Link rest of video chain
        compositor.link(compositor_caps)
        compositor_caps.link(post_compositor_convert)
        post_compositor_convert.link(self.text_overlay)
        self.text_overlay.link(x264enc)
        x264enc.link(h264parse)
        h264parse.link(video_mux_queue)

        # Link audio chain
        self.appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audio_caps_stereo)
        audio_caps_stereo.link(audio_encoder)
        audio_encoder.link(aacparse)
        aacparse.link(audio_mux_queue)

        # Link mux to sink
        flvmux.link(queue)
        queue.link(rtmpsink)

        # Connect signals
        self.appsrc.connect("need-data", self._on_need_data)
        self.waveform_appsrc.connect("need-data", self._on_waveform_need_data)

        # Initialize push functions
        GLib.timeout_add(1, self._push_audio_data)  # Check every 1ms for precise timing
        GLib.timeout_add(33, self._push_waveform_data)

        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ Fallback pipeline configured with waveform")

    def _on_need_data(self, src, length):
        """Handle when appsrc needs data"""
        # This will be handled by the periodic push - removed debug log
        pass

    def _on_waveform_need_data(self, src, length):
        """Handle when waveform appsrc needs data"""
        # This will be handled by the periodic push
        pass

    def _on_enough_data(self, src):
        """Handle when appsrc has enough data"""
        logger.info("[AUDIO] GStreamer signaling enough-data - buffer full")

    def _push_audio_data(self):
        """Push audio data with quality-focused adaptive handling"""
        if not self.is_streaming:
            return False

        # Track timing
        current_time = time.time()
        if not hasattr(self, "_last_audio_push_time"):
            self._last_audio_push_time = current_time
            self._push_count = 0
            self._last_log_time = current_time
            self._consecutive_pushes = 0
            self._audio_pushed_time = 0  # Track total audio time pushed
            self._current_speed_target = 1.0  # Track gradual speed changes
            self._burst_mode = False  # Track if we're in burst recovery
            self._last_incoming_rate = 0.0

        # Get buffer health to determine push rate
        buffer_health = self.audio_mixer.get_buffer_health()
        buffer_seconds = buffer_health['seconds']
        
        # Get incoming rate for burst detection
        incoming_rate = getattr(self.audio_mixer, 'last_incoming_rate', 0.0)
        
        # Detect burst conditions
        if incoming_rate > 3.0 and not self._burst_mode:
            self._burst_mode = True
            logger.info(f"[AUDIO] BURST DETECTED: {incoming_rate:.1f}x incoming rate - entering burst mode")
        elif incoming_rate < 1.5 and self._burst_mode and buffer_seconds < 3.0:
            self._burst_mode = False
            logger.info(f"[AUDIO] Burst ended, returning to normal mode")
        
        # ALWAYS use 20ms chunks
        chunk_duration = 0.02  # Always 20ms chunks
        base_interval = chunk_duration  # Base interval same as chunk duration

        # Quality-focused adaptive system with conservative speeds
        if buffer_seconds > 25.0:
            # EMERGENCY: Buffer too large - clear it
            logger.warning(f"[AUDIO] Emergency buffer clear at {buffer_seconds:.1f}s")
            self.audio_mixer.clear_buffer()
            target_speed = 1.0
            mode = "emergency"
            self._burst_mode = False
        elif buffer_seconds > 15.0:
            # VERY HIGH: Large buffer - moderate catch-up only
            # Max 1.35x to maintain quality
            target_speed = 1.35
            mode = "very-high"
            
            if self._consecutive_pushes == 0:
                logger.info(f"[AUDIO] Buffer very high ({buffer_seconds:.1f}s), quality-preserving catch-up")
        elif buffer_seconds > 10.0:
            # HIGH: Buffer large - gentle catch-up
            # Max 1.2x for good quality
            target_speed = 1.2
            mode = "high"
                
            if self._consecutive_pushes == 0:
                logger.info(f"[AUDIO] Buffer high ({buffer_seconds:.1f}s), gentle catch-up mode")
        elif buffer_seconds > 4.0:
            # MODERATE: Buffer growing - very gentle catch-up
            # Max 1.1x - barely noticeable
            target_speed = 1.1
            mode = "moderate"
                
            if self._consecutive_pushes == 0:
                logger.info(f"[AUDIO] Buffer moderate ({buffer_seconds:.1f}s), subtle catch-up mode")
        elif buffer_seconds > 1.5:
            # SLIGHT: Buffer building - minimal speedup
            target_speed = 1.02  # Very subtle
            mode = "slight"
            self._consecutive_pushes = 0
        elif buffer_seconds > 0.3:
            # NORMAL: Buffer healthy
            target_speed = 1.0  # Real-time
            mode = "normal"
            self._consecutive_pushes = 0
        else:
            # LOW: Buffer too low - slow down slightly to prevent underrun
            target_speed = 0.98  # Slight slowdown
            mode = "low"
            self._consecutive_pushes = 0

        # Very gradual speed transitions for quality
        if not hasattr(self, '_current_speed_target'):
            self._current_speed_target = 1.0
            
        # Slow transitions to avoid jarring speed changes
        max_change = 0.01  # Very gradual changes (was 0.02-0.05)
        
        # Gradually move toward target speed
        speed_diff = target_speed - self._current_speed_target
        
        if abs(speed_diff) > max_change:
            if speed_diff > 0:
                self._current_speed_target += max_change
            else:
                self._current_speed_target -= max_change
        else:
            self._current_speed_target = target_speed

        # Calculate interval based on current speed
        min_interval = base_interval / self._current_speed_target

        # Track mode changes
        if mode not in ["normal", "low"]:
            self._consecutive_pushes += 1
        
        # Calculate time since last push
        time_elapsed = current_time - self._last_audio_push_time

        # Enforce minimum interval
        if time_elapsed < min_interval:
            return True  # Too early, skip this cycle

        self._last_audio_push_time = current_time
        self._push_count += 1
        self._audio_pushed_time += chunk_duration  # Use chunk_duration for accurate tracking

        # Push audio chunk - ALWAYS same size and aligned
        base_chunk_size = int(config.AUDIO_SAMPLE_RATE * chunk_duration * config.AUDIO_CHANNELS * 2)
        # Ensure chunk size is aligned to 2-byte boundaries for int16 samples
        chunk_size = (base_chunk_size // 2) * 2
        if chunk_size == 0:
            chunk_size = 2  # Minimum one sample
        
        # Log every second
        if current_time - self._last_log_time >= 1.0:
            effective_rate = self._push_count / (current_time - self._last_log_time)
            real_time_elapsed = current_time - self._last_log_time  # Calculate real time elapsed
            playback_speed = self._audio_pushed_time / real_time_elapsed if real_time_elapsed > 0 else 1.0
            
            # Add incoming rate info if available
            incoming_info = ""
            if hasattr(self.audio_mixer, 'last_incoming_rate'):
                incoming_info = f", In: {self.audio_mixer.last_incoming_rate:.1f}x"
            
            burst_info = " [BURST]" if self._burst_mode else ""
            
            # Warn if speed is too high
            speed_warning = ""
            if playback_speed > 1.3:
                speed_warning = " ⚠️ HIGH SPEED"
            
            logger.info(f"[AUDIO] Push stats - Rate: {self._push_count} pushes/sec, Chunk: {chunk_size} bytes, Mixer: {buffer_seconds:.2f}s ({buffer_health['status']}), Mode: {mode}{burst_info}, Speed: {playback_speed:.2f}x (target: {self._current_speed_target:.2f}x){incoming_info}{speed_warning}")
            self._push_count = 0
            self._last_log_time = current_time
            self._audio_pushed_time = 0  # Reset for next period
            self._last_incoming_rate = incoming_rate
        
        audio_data = self.audio_mixer.get_audio_chunk(chunk_size)

        if audio_data:
            # Additional safety check for buffer alignment
            if len(audio_data) % 2 != 0:
                logger.error(f"[AUDIO] Received unaligned audio data: {len(audio_data)} bytes")
                # Truncate to aligned size
                audio_data = audio_data[:(len(audio_data) // 2) * 2]
            
            if audio_data:  # Check again after potential truncation
                buffer = Gst.Buffer.new_wrapped(audio_data)
                ret = self.appsrc.emit("push-buffer", buffer)
                if ret != Gst.FlowReturn.OK:
                    logger.warning(f"[AUDIO] Push failed: {ret}")
                # Removed debug log for successful pushes to reduce spam

        return True  # Continue calling

    def _push_waveform_data(self):
        """Push waveform visualization data with smooth random animation"""
        if not self.is_streaming or not self.waveform_appsrc:
            logger.debug(f"[WAVEFORM] Not pushing - streaming: {self.is_streaming}, appsrc: {self.waveform_appsrc is not None}")
            return False

        # Simple mouth-like animation based on whether audio is playing
        if self.audio_mixer.is_speaking:
            # Audio is playing - create random mouth opening/closing animation
            if not hasattr(self, '_waveform_time'):
                self._waveform_time = 0
            
            # Create random talking mouth movement
            time_factor = self._waveform_time * self.waveform_animation_speed
            
            # Generate random mouth openness (like talking)
            # Use multiple sine waves to create natural talking rhythm
            primary_talk = np.sin(time_factor) * 0.45  # Increased amplitude
            secondary_talk = np.sin(time_factor * 1.8) * 0.35  # Increased amplitude
            quick_flutter = np.sin(time_factor * 3.5) * 0.20  # Increased amplitude
            micro_variations = np.sin(time_factor * 5.2) * 0.12  # Increased amplitude
            
            # Add sentence pauses - periods of silence like natural speech
            # Create longer wave that occasionally goes negative for silence
            sentence_pattern = np.sin(time_factor * 0.3) * 0.8  # Very slow wave for sentences
            pause_pattern = np.sin(time_factor * 0.15) * 0.5   # Even slower for longer pauses
            
            # Combine sentence patterns - when negative, create silence
            sentence_factor = sentence_pattern + pause_pattern
            
            # If sentence factor is negative, create silence (pause between sentences)
            if sentence_factor < -0.3:
                # Silence period - no mouth movement
                self.waveform_target_openness = 0.0
            else:
                # Speaking period - combine for natural talking motion with more variation
                base_openness = 0.35 + sentence_factor * 0.2  # Higher base with sentence modulation
                self.waveform_target_openness = base_openness + primary_talk + secondary_talk + quick_flutter + micro_variations
            
            # Keep mouth openness in reasonable bounds (0.0 to 1.0 for bigger amplitude)
            self.waveform_target_openness = max(0.0, min(1.0, self.waveform_target_openness))
            
            # Smooth interpolation to target openness
            self.waveform_mouth_openness = (self.waveform_mouth_openness * 0.7 + 
                                          self.waveform_target_openness * 0.3)
            
            self._waveform_time += 2.0  # Faster time progression for more dynamic talking
            
        else:
            # No audio - close mouth smoothly
            self.waveform_target_openness = 0.0
            self.waveform_mouth_openness *= 0.85  # Smooth closing
            if self.waveform_mouth_openness < 0.02:
                self.waveform_mouth_openness = 0.0

        # Create single-value array for mouth openness (the generator expects an array)
        mouth_level = np.array([self.waveform_mouth_openness])
        
        # Update stored levels with mouth openness
        with self.waveform_lock:
            self.waveform_levels = mouth_level

        # Create waveform overlay
        overlay_data = self.waveform_generator.create_waveform_overlay(self.waveform_levels)

        # Create buffer
        buffer = Gst.Buffer.new_wrapped(overlay_data)
        # Let GStreamer handle timestamps since do-timestamp=True

        ret = self.waveform_appsrc.emit("push-buffer", buffer)
        if ret != Gst.FlowReturn.OK:
            logger.warning(f"Waveform push failed: {ret}")
        # Removed verbose waveform debug logging to reduce spam

        return True  # Continue calling

    def _on_bus_message(self, bus, message):
        """Handle bus messages"""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logger.error(
                f"Pipeline error from {message.src.get_name()}: {error.message}"
            )
            logger.error(f"Debug: {debug}")
            self.stop_streaming()
        elif msg_type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            logger.warning(
                f"Pipeline warning from {message.src.get_name()}: {warning.message}"
            )
            logger.warning(f"Debug: {debug}")
        elif msg_type == Gst.MessageType.EOS:
            # Handle end of stream - loop the video
            logger.info("End of stream detected, looping video...")
            self.pipeline.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            )
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info(f"Pipeline state: {old.value_nick} -> {new.value_nick}")
        elif msg_type == Gst.MessageType.BUFFERING:
            percent = message.parse_buffering()
            logger.info(f"Buffering: {percent}%")
        elif msg_type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure and structure.get_name() == "rtmpsink-stats":
                logger.info(f"RTMP stats: {structure.to_string()}")

        return True

    def initialize_realtime_api(self) -> None:
        """Initialize Realtime API connection"""
        try:
            logger.info("Initializing Realtime API...")
            self.loop_thread = threading.Thread(
                target=self.start_event_loop, daemon=True
            )
            self.loop_thread.start()

            # Wait for connection
            time.sleep(2)

            if self.realtime_client and self.realtime_client.is_connected:
                logger.info("✅ Realtime API ready")
            else:
                logger.warning("Realtime API not connected, continuing without it")
                self.realtime_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Realtime API: {e}")
            self.realtime_client = None

    def send_message_to_realtime(self, message: str, author: str):
        """Send message for response"""
        if (
            self.event_loop
            and self.realtime_client
            and self.realtime_client.is_connected
        ):
            try:
                # Only clear buffer if it's extremely large (>12s)
                # The conservative speed system should handle smaller buffers gracefully
                buffer_health = self.audio_mixer.get_buffer_health()
                if buffer_health["seconds"] > 12.0:
                    logger.info(
                        f"Large buffer detected ({buffer_health['seconds']:.1f}s), clearing old audio"
                    )
                    self.audio_mixer.clear_buffer()
                elif buffer_health["seconds"] > 6.0:
                    logger.info(
                        f"Moderate buffer detected ({buffer_health['seconds']:.1f}s), conservative speed will handle"
                    )

                asyncio.run_coroutine_threadsafe(
                    self.realtime_client.send_text_message(message, author),
                    self.event_loop,
                )
            except Exception as e:
                logger.error(f"Error sending to Realtime API: {e}")
        else:
            logger.info(f"[No Realtime API] {author}: {message}")

    def update_text_overlay(self, text: str) -> None:
        """Update text overlay"""
        if self.text_overlay and text:
            if len(text) > 80:
                text = text[:77] + "..."
            self.text_overlay.set_property("text", text)

    def monitor_youtube_chat(self) -> None:
        """Safe & Fast YouTube chat monitoring"""
        logger.info("💬 Starting SAFE & FAST YouTube chat monitoring...")

        if not self.youtube_api.live_chat_id:
            logger.error("❌ No live chat ID available")
            return

        logger.info(f"✅ Monitoring chat: {self.youtube_api.live_chat_id}")

        next_page_token = None
        consecutive_empty_responses = 0
        last_request_time = 0
        min_request_interval = 0.5  # 500ms minimum - SAFE LIMIT

        # Adaptive polling rate
        current_poll_rate = 0.5  # Start conservative

        while self.is_streaming:
            try:
                # Enforce minimum interval
                time_since_last = time.time() - last_request_time
                if time_since_last < min_request_interval:
                    time.sleep(min_request_interval - time_since_last)

                last_request_time = time.time()

                # Make request
                messages = self.youtube_api.get_chat_messages_fast(next_page_token)

                if messages and "messages" in messages:
                    new_messages = 0

                    for msg in messages["messages"]:
                        msg_hash = hash(f"{msg['timestamp']}{msg['message'][:10]}")

                        if msg_hash not in self.message_hashes:
                            self.message_hashes.add(msg_hash)
                            new_messages += 1

                            # Process message
                            username = msg["author"]
                            message = msg["message"]

                            logger.info(f"{username}: {message}")
                            self.update_text_overlay(f"{username}: {message}")
                            self.send_message_to_realtime(message, username)

                    # Auto-cleanup message set
                    if len(self.message_hashes) > 1000:
                        self.message_hashes = set(list(self.message_hashes)[-500:])

                    next_page_token = messages.get("nextPageToken")

                    # RESPECT YouTube's polling interval suggestion
                    youtube_suggested = (
                        messages.get("pollingIntervalMillis", 2000) / 1000.0
                    )

                    # Adaptive rate based on activity
                    if new_messages > 0:
                        consecutive_empty_responses = 0
                        # Active chat: Poll faster but respect YouTube's minimum
                        current_poll_rate = max(
                            min_request_interval, youtube_suggested * 0.5
                        )
                    else:
                        consecutive_empty_responses += 1
                        # Slow down if no new messages
                        if consecutive_empty_responses > 5:
                            current_poll_rate = min(youtube_suggested, 2.0)
                        else:
                            current_poll_rate = youtube_suggested * 0.75

                    logger.debug(
                        f"Next poll in {current_poll_rate:.1f}s (YouTube suggests {youtube_suggested:.1f}s)"
                    )
                    time.sleep(current_poll_rate)

                else:
                    # No messages - conservative wait
                    consecutive_empty_responses += 1
                    time.sleep(1.0)

            except Exception as e:
                if "quotaExceeded" in str(e) or "rateLimitExceeded" in str(e):
                    logger.error("⚠️  RATE LIMIT HIT! Backing off for 60 seconds")
                    time.sleep(60)  # Long backoff to avoid ban
                elif "forbidden" in str(e).lower():
                    logger.error("❌ FORBIDDEN - Possible ban! Stopping chat monitor")
                    break
                else:
                    logger.error(f"Chat error: {e}")
                    time.sleep(2)  # Safe recovery time

    def monitor_twitch_chat(self) -> None:
        """Monitor simulated Twitch chat"""
        logger.info("💬 Starting Twitch chat simulation...")

        time.sleep(3)
        welcome_msg = "What's up!"
        self.update_text_overlay(f"{welcome_msg}")
        self.send_message_to_realtime(welcome_msg, "System")

        last_activity = time.time()

        while self.is_streaming:
            try:
                sim_msg = self.twitch_chat_sim.get_next_message()

                if sim_msg:
                    msg_id = f"{sim_msg['author']}_{sim_msg['timestamp']}_{sim_msg['message'][:20]}"

                    if msg_id not in self.processed_messages:
                        self.processed_messages.append(msg_id)

                        username = sim_msg["author"]
                        message = sim_msg["message"]

                        logger.info(f"💬 [Twitch] {username}: {message}")
                        self.update_text_overlay(f"{username}: {message}")
                        self.send_message_to_realtime(message, username)

                        last_activity = time.time()

                # Activity check
                if time.time() - last_activity > 45:  # Increased from 25 to 45 seconds
                    activity_msg = random.choice(
                        [
                            "PogChamp!",
                            "Let's go chat!",
                            "Drop a Kappa!",
                            "Hype in chat!",
                        ]
                    )
                    self.update_text_overlay(f"{activity_msg}")
                    self.send_message_to_realtime(activity_msg, "System")
                    last_activity = time.time()

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Twitch chat error: {e}")
                time.sleep(2)

    def start_youtube_streaming(self, use_existing_broadcast: str = None) -> None:
        """Start YouTube streaming"""
        try:
            self.platform = "youtube"
            self.is_streaming = True
            self.waveform_running = True

            # Initialize YouTube API
            self.youtube_api = YouTubeAPIManager(
                config.YOUTUBE_CLIENT_ID,
                config.YOUTUBE_CLIENT_SECRET,
                config.YOUTUBE_REFRESH_TOKEN,
                config.YOUTUBE_SCOPES
            )

            # Initialize Realtime API
            self.initialize_realtime_api()

            # Start buffer monitor
            self.buffer_monitor_thread = threading.Thread(
                target=self.monitor_audio_buffer, daemon=True
            )
            self.buffer_monitor_thread.start()

            # YouTube setup
            self.youtube_api.authenticate()

            if use_existing_broadcast:
                logger.info(f"Using existing broadcast: {use_existing_broadcast}")
                self.youtube_api.broadcast_id = use_existing_broadcast

                broadcast = (
                    self.youtube_api.youtube.liveBroadcasts()
                    .list(part="snippet,status", id=use_existing_broadcast)
                    .execute()
                )

                if broadcast["items"]:
                    item = broadcast["items"][0]
                    self.youtube_api.live_chat_id = item["snippet"].get("liveChatId")
                    stream_key = input("Enter YouTube stream key: ").strip()
                else:
                    raise Exception("Broadcast not found")
            else:
                self.youtube_api.create_broadcast()
                stream_key = self.youtube_api.create_stream(config.VIDEO_HEIGHT, config.VIDEO_FRAMERATE)
                self.youtube_api.bind_broadcast_to_stream()

            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)

            # Start streaming
            logger.info("▶️  Starting YouTube stream...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.info("Pipeline starting asynchronously...")
                ret, state, pending = self.pipeline.get_state(30 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.SUCCESS:
                    logger.info(
                        f"Pipeline started successfully, state: {state.value_nick}"
                    )
                elif ret == Gst.StateChangeReturn.ASYNC:
                    logger.warning(
                        "Pipeline still changing state, continuing anyway..."
                    )
                else:
                    raise Exception(f"Pipeline failed to start: {ret}")
            elif ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("Pipeline started successfully")

            # Give pipeline time to stabilize
            time.sleep(2)

            # Generate initial audio AND video data to activate stream
            # Use clear_buffer first to ensure we don't trigger speech detection
            self.audio_mixer.clear_buffer()
            initial_silence = bytes(
                int(config.AUDIO_SAMPLE_RATE * 1.0 * config.AUDIO_CHANNELS * 2)
            )  # 1 second
            # Add silence directly without triggering speech detection
            with self.audio_mixer.buffer_lock:
                self.audio_mixer.audio_buffer.append(initial_silence)
                self.audio_mixer.total_bytes_buffered += len(initial_silence)
                # Don't set is_speaking to True
            logger.info("Added initial audio to start stream")

            # Force push some audio data immediately
            for _ in range(10):
                self._push_audio_data()
                time.sleep(0.02)

            # Wait for stream to be active with longer timeout
            if self.youtube_api.wait_for_stream_active(timeout=60):
                # Try to transition to live, but handle if already live
                try:
                    self.youtube_api.transition_to_live()
                except Exception as e:
                    logger.warning(f"Transition warning (may already be live): {e}")
                    # Try to get the live chat ID anyway
                    try:
                        broadcast = (
                            self.youtube_api.youtube.liveBroadcasts()
                            .list(
                                part="snippet,status", id=self.youtube_api.broadcast_id
                            )
                            .execute()
                        )

                        if broadcast["items"]:
                            item = broadcast["items"][0]
                            status = item["status"]["lifeCycleStatus"]
                            logger.info(f"Broadcast status: {status}")

                            if status == "live":
                                self.youtube_api.live_chat_id = item["snippet"].get(
                                    "liveChatId"
                                )
                                logger.info(
                                    f"✅ Broadcast is live! Chat ID: {self.youtube_api.live_chat_id}"
                                )
                            else:
                                logger.error(f"Unexpected broadcast status: {status}")
                                return
                    except Exception as fetch_error:
                        logger.error(f"Failed to fetch broadcast info: {fetch_error}")
                        return

                # Start chat monitoring
                chat_thread = threading.Thread(
                    target=self.monitor_youtube_chat, daemon=True
                )
                chat_thread.start()

                logger.info("\n" + "=" * 70)
                logger.info(
                    "✅ ULTRA-FAST YOUTUBE STREAMING WITH CLEAN AUDIO & SMOOTH WAVEFORM"
                )
                logger.info(
                    f"📺 Watch: https://youtube.com/watch?v={self.youtube_api.broadcast_id}"
                )
                logger.info(f"🎬 Video: {config.BACKGROUND_VIDEO_PATH}")
                logger.info(f"🔊 Voice: {config.OPENAI_VOICE} (Realtime API)")
                logger.info("🌊 Smooth audio waveform visualization (no freezing)")
                logger.info("💬 Safe & Fast YouTube chat monitoring")
                logger.info("🎵 Clean audio playback with noise reduction")
                logger.info("=" * 70 + "\n")

                # Main loop
                loop = GLib.MainLoop()
                loop.run()
            else:
                logger.error("Stream activation timeout")

        except Exception as e:
            logger.error(f"YouTube streaming error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop_streaming()

    def start_twitch_streaming(self, stream_key: str) -> None:
        """Start Twitch streaming"""
        try:
            self.platform = "twitch"
            self.is_streaming = True
            self.waveform_running = True

            # Initialize Twitch chat simulator
            self.twitch_chat_sim = TwitchChatSimulator(config.TWITCH_SIMULATED_MESSAGES)

            # Initialize Realtime API
            self.initialize_realtime_api()

            # Start buffer monitor
            self.buffer_monitor_thread = threading.Thread(
                target=self.monitor_audio_buffer, daemon=True
            )
            self.buffer_monitor_thread.start()

            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)

            # Start streaming
            logger.info("▶️  Starting Twitch stream...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.info("Pipeline starting asynchronously...")
                # Shorter timeout for Twitch
                ret, state, pending = self.pipeline.get_state(10 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.SUCCESS:
                    logger.info(
                        f"Pipeline started successfully, state: {state.value_nick}"
                    )
                elif ret == Gst.StateChangeReturn.ASYNC:
                    logger.warning(
                        "Pipeline still changing state, continuing anyway..."
                    )
                    # Force playing state
                    self.pipeline.set_state(Gst.State.PLAYING)
                else:
                    raise Exception(f"Pipeline failed to start: {ret}")
            elif ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("Pipeline started successfully")

            # Give pipeline time to stabilize
            time.sleep(2)

            # Generate initial audio
            # Use clear_buffer first to ensure we don't trigger speech detection
            self.audio_mixer.clear_buffer()
            initial_silence = bytes(
                int(config.AUDIO_SAMPLE_RATE * 0.5 * config.AUDIO_CHANNELS * 2)
            )  # 0.5 seconds
            # Add silence directly without triggering speech detection
            with self.audio_mixer.buffer_lock:
                self.audio_mixer.audio_buffer.append(initial_silence)
                self.audio_mixer.total_bytes_buffered += len(initial_silence)
                # Don't set is_speaking to True
            logger.info("Added initial audio to start stream")

            # Start chat monitoring
            chat_thread = threading.Thread(target=self.monitor_twitch_chat, daemon=True)
            chat_thread.start()

            logger.info("\n" + "=" * 70)
            logger.info(
                "✅ ULTRA-LOW LATENCY TWITCH STREAMING WITH CLEAN AUDIO & SMOOTH WAVEFORM"
            )
            logger.info("📺 Check your Twitch channel")
            logger.info(f"🎬 Video: {config.BACKGROUND_VIDEO_PATH}")
            logger.info(f"🔊 Voice: {config.OPENAI_VOICE} (Realtime API)")
            logger.info("🌊 Smooth audio waveform visualization (no freezing)")
            logger.info("💬 Simulated chat responses")
            logger.info("🎵 Clean audio playback with noise reduction")
            logger.info("=" * 70 + "\n")

            # Main loop
            loop = GLib.MainLoop()
            loop.run()

        except Exception as e:
            logger.error(f"Twitch streaming error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop_streaming()

    def stop_streaming(self) -> None:
        """Stop streaming"""
        self.is_streaming = False
        self.waveform_running = False

        # Stop Realtime API
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Stop YouTube broadcast
        if self.platform == "youtube" and self.youtube_api and hasattr(self.youtube_api, "broadcast_id"):
            self.youtube_api.stop_broadcast()

        logger.info("✅ Stream stopped")

    def run(
        self, platform: str, broadcast_id: str = None, stream_key: str = None
    ) -> None:
        """Run the streamer"""
        try:
            print("\n" + "=" * 70)
            print("    🚀 ULTRA-FAST AI STREAMER WITH CLEAN AUDIO & SMOOTH WAVEFORM")
            print("    ⚡ OpenAI Realtime API")
            print("    🏎️  Safe & Fast chat monitoring")
            print("    🎵 Clean audio playback (no noise)")
            print("    🎬 Video background support")
            print("    🌊 Smooth waveform visualization (no freezing)")
            print("=" * 70)
            print(f"📺 Platform: {platform.upper()}")
            print(f"🎥 Video: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}@{config.VIDEO_FRAMERATE}fps")
            print(f"🔊 Audio: 24kHz Realtime ({config.OPENAI_VOICE})")
            print(f"📁 Background: {config.BACKGROUND_VIDEO_PATH}")
            print(
                f"🌊 Waveform: {config.WAVEFORM_WIDTH}x{config.WAVEFORM_HEIGHT} ({config.WAVEFORM_BARS} bars)"
            )
            print("=" * 70 + "\n")

            if platform == "youtube":
                self.start_youtube_streaming(use_existing_broadcast=broadcast_id)
            elif platform == "twitch":
                if not stream_key:
                    stream_key = input("Twitch stream key: ").strip()
                self.start_twitch_streaming(stream_key)
            else:
                logger.error(f"Unknown platform: {platform}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1) 