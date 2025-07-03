"""
Dual Robot Live AI Streamer
Manages streaming with two robots having conversations
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
from ..audio import DualAudioMixer, DualWaveformGenerator
from ..api import DualRobotRealtimeManager
from ..platforms import YouTubeAPIManager, TwitchChatSimulator

logger = logging.getLogger(__name__)


class DualRobotLiveAIStreamer:
    """Live AI streamer with dual robot conversation system"""

    def __init__(self):
        self.platform = None
        self.youtube_api = None
        self.twitch_chat_sim = None
        self.dual_robot_manager = None
        self.pipeline = None
        self.is_streaming = False
        
        # Audio system
        self.audio_mixer = DualAudioMixer()
        self.appsrc = None
        
        # Waveform system
        self.waveform_generator = DualWaveformGenerator(
            width=config.WAVEFORM_WIDTH,
            height=config.WAVEFORM_HEIGHT,
            bars=config.WAVEFORM_BARS
        )
        self.waveform_appsrc = None
        self.waveform_timestamp = 0
        self.waveform_update_thread = None
        self.waveform_running = False
        
        # Text overlay
        self.text_overlay = None
        
        # Chat monitoring
        self.processed_messages = deque(maxlen=500)
        self.message_hashes = set()
        
        # Audio timestamps
        self.audio_timestamp = 0
        self.last_audio_time = time.time()
        
        # Initialize timestamps properly
        self.start_time = time.time()
        self.audio_timestamp = 0
        self.waveform_timestamp = 0
        
        # Buffer monitoring
        self.buffer_monitor_thread = None
        
        logger.info("[DUAL_STREAMER] Dual robot streamer initialized")

    def left_robot_audio_callback(self, audio_data: bytes):
        """Callback for left robot audio"""
        self.audio_mixer.add_left_audio(audio_data)
        self.waveform_generator.update_left_levels(audio_data)
        self.last_audio_time = time.time()

    def right_robot_audio_callback(self, audio_data: bytes):
        """Callback for right robot audio"""
        self.audio_mixer.add_right_audio(audio_data)
        self.waveform_generator.update_right_levels(audio_data)
        self.last_audio_time = time.time()

    def left_robot_level_callback(self, audio_data: bytes):
        """Callback for left robot audio levels (for waveform)"""
        self.waveform_generator.update_left_levels(audio_data)

    def right_robot_level_callback(self, audio_data: bytes):
        """Callback for right robot audio levels (for waveform)"""
        self.waveform_generator.update_right_levels(audio_data)

    def check_video_file(self) -> bool:
        """Check if background video file exists"""
        if not os.path.exists(config.BACKGROUND_VIDEO_PATH):
            logger.error(f"Background video not found: {config.BACKGROUND_VIDEO_PATH}")
            return False
        return True

    def setup_gstreamer_pipeline(self, stream_key: str) -> None:
        """Setup GStreamer pipeline for dual robot streaming"""
        if not self.check_video_file():
            raise FileNotFoundError("Background video file not found")

        # Use the correct RTMP URL based on platform
        if self.platform == "twitch":
            rtmp_url = f"{config.TWITCH_RTMP_URL}/{stream_key}"
        else:  # YouTube
            rtmp_url = f"{config.YOUTUBE_RTMP_BASE_URL}/{stream_key}"
            
        logger.info(f"Setting up pipeline for {self.platform.upper()} RTMP: {rtmp_url}")

        # Create simplified pipeline - first try without compositor to isolate issues
        pipeline_str = f"""
        filesrc location="{config.BACKGROUND_VIDEO_PATH}" ! 
        decodebin !
        videoconvert ! 
        videoscale ! 
        video/x-raw,format=I420,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},framerate={config.VIDEO_FRAMERATE}/1 !
        x264enc bitrate={config.VIDEO_BITRATE} tune=zerolatency speed-preset=ultrafast threads=4 !
        video/x-h264,profile=baseline !
        h264parse ! 
        mux. 
        
        appsrc name=audiosrc 
            caps="audio/x-raw,format=S16LE,rate={config.AUDIO_SAMPLE_RATE},channels={config.AUDIO_CHANNELS}"
            stream-type=0 max-bytes=200000 block=false 
            format=time is-live=true do-timestamp=false !
        audioconvert ! 
        audioresample ! 
        voaacenc bitrate={config.AUDIO_BITRATE} !
        aacparse ! 
        mux.
        
        flvmux name=mux streamable=true !
        rtmpsink location="{rtmp_url}" sync=false async=false
        """
        
        # Note: Temporarily removed waveform overlay to isolate pipeline issues
        # We'll add it back once basic streaming works

        logger.info("Creating GStreamer pipeline...")
        logger.debug(f"Pipeline string: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        if not self.pipeline:
            raise RuntimeError("Failed to create GStreamer pipeline")

        # Get elements
        self.appsrc = self.pipeline.get_by_name("audiosrc")
        self.waveform_appsrc = self.pipeline.get_by_name("waveformsrc")  # Will be None for now

        if not self.appsrc:
            raise RuntimeError("Failed to get audio appsrc element")

        # Set up callbacks
        self.appsrc.connect("need-data", self._on_need_data)
        self.appsrc.connect("enough-data", self._on_enough_data)
        
        # Only connect waveform callback if element exists
        if self.waveform_appsrc:
            self.waveform_appsrc.connect("need-data", self._on_waveform_need_data)

        # Set up message bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("✅ GStreamer pipeline setup complete")

    def _on_need_data(self, src, length):
        """Audio data callback - pipeline is ready for audio data"""
        if not self.is_streaming:
            return
        logger.debug(f"[DUAL_STREAMER] Pipeline requesting audio data, length: {length}")
        self._push_audio_data()

    def _on_waveform_need_data(self, src, length):
        """Waveform data callback"""
        if not self.is_streaming:
            return
        threading.Thread(target=self._push_waveform_data, daemon=True).start()

    def _on_enough_data(self, src):
        """Stop pushing data when enough is buffered"""
        pass

    def _push_audio_data(self):
        """Push mixed audio data to pipeline"""
        try:
            chunk_size = int(config.AUDIO_SAMPLE_RATE * 0.02 * 2)  # 20ms chunks
            audio_chunk = self.audio_mixer.get_audio_chunk(chunk_size)
            
            if audio_chunk and len(audio_chunk) > 0:
                buffer = Gst.Buffer.new_allocate(None, len(audio_chunk), None)
                buffer.fill(0, audio_chunk)
                
                # Let GStreamer handle timestamps for now
                buffer.pts = Gst.CLOCK_TIME_NONE
                buffer.duration = Gst.CLOCK_TIME_NONE
                
                # Push to pipeline
                ret = self.appsrc.emit("push-buffer", buffer)
                if ret != Gst.FlowReturn.OK:
                    logger.warning(f"[DUAL_STREAMER] Audio push failed: {ret}")
                    # If the pipeline is not consuming audio properly, log it
                    if ret == Gst.FlowReturn.FLUSHING:
                        logger.warning("[DUAL_STREAMER] Pipeline is flushing - may not be in PLAYING state")
            else:
                # Push silence if no audio available
                silence = b'\x00' * chunk_size
                buffer = Gst.Buffer.new_allocate(None, len(silence), None)
                buffer.fill(0, silence)
                buffer.pts = Gst.CLOCK_TIME_NONE
                buffer.duration = Gst.CLOCK_TIME_NONE
                ret = self.appsrc.emit("push-buffer", buffer)
                if ret != Gst.FlowReturn.OK:
                    logger.debug(f"[DUAL_STREAMER] Silence push result: {ret}")

        except Exception as e:
            logger.error(f"[DUAL_STREAMER] Error pushing audio: {e}")

    def _push_waveform_data(self):
        """Push waveform overlay data to pipeline"""
        # Skip if waveform element not present
        if not self.waveform_appsrc:
            return
            
        try:
            # Animate waveforms
            self.waveform_generator.animate_waveforms()
            
            # Create video frame for overlay
            frame = np.zeros((config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 4), dtype=np.uint8)
            
            # Get robot positions from config
            left_pos = (config.ROBOT_POSITIONS["left"]["x"], config.ROBOT_POSITIONS["left"]["y"])
            right_pos = (config.ROBOT_POSITIONS["right"]["x"], config.ROBOT_POSITIONS["right"]["y"])
            
            # Generate combined waveform overlay
            overlay = self.waveform_generator.get_combined_waveform_overlay(
                frame, left_pos, right_pos
            )
            
            # Convert to bytes
            frame_bytes = overlay.tobytes()
            
            # Create GStreamer buffer
            buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
            buffer.fill(0, frame_bytes)
            
            # Set timestamp
            buffer.pts = self.waveform_timestamp
            buffer.duration = Gst.SECOND // config.VIDEO_FRAMERATE
            self.waveform_timestamp += buffer.duration
            
            # Push to pipeline
            ret = self.waveform_appsrc.emit("push-buffer", buffer)
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"[DUAL_STREAMER] Waveform push failed: {ret}")

        except Exception as e:
            logger.error(f"[DUAL_STREAMER] Error pushing waveform: {e}")

    def _push_initial_data(self):
        """Push initial data to appsrc elements to help pipeline start"""
        try:
            # Initialize timestamps
            self.audio_timestamp = 0
            self.waveform_timestamp = 0
            
            # Push initial silence for audio
            chunk_size = int(config.AUDIO_SAMPLE_RATE * 0.02 * 2)  # 20ms
            silence = b'\x00' * chunk_size
            
            audio_buffer = Gst.Buffer.new_allocate(None, len(silence), None)
            audio_buffer.fill(0, silence)
            audio_buffer.pts = self.audio_timestamp
            audio_buffer.duration = Gst.SECOND * len(silence) // (config.AUDIO_SAMPLE_RATE * 2)
            self.audio_timestamp += audio_buffer.duration
            
            self.appsrc.emit("push-buffer", audio_buffer)
            logger.debug("[DUAL_STREAMER] Pushed initial audio silence")
            
            # Push initial transparent frame for waveform (if element exists)
            if self.waveform_appsrc:
                frame = np.zeros((config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 4), dtype=np.uint8)
                frame_bytes = frame.tobytes()
                
                waveform_buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
                waveform_buffer.fill(0, frame_bytes)
                waveform_buffer.pts = self.waveform_timestamp
                waveform_buffer.duration = Gst.SECOND // config.VIDEO_FRAMERATE
                self.waveform_timestamp += waveform_buffer.duration
                
                self.waveform_appsrc.emit("push-buffer", waveform_buffer)
                logger.debug("[DUAL_STREAMER] Pushed initial waveform frame")
            else:
                logger.debug("[DUAL_STREAMER] Skipping waveform - element not present")
            
        except Exception as e:
            logger.error(f"[DUAL_STREAMER] Error pushing initial data: {e}")

    def _test_rtmp_connection(self, stream_key: str):
        """Test RTMP connection independently"""
        try:
            import subprocess
            import sys
            
            # Test with a simple GStreamer pipeline
            rtmp_url = f"rtmp://bog01.contribute.live-video.net/app/{stream_key}"
            
            # Create minimal test pipeline
            test_pipeline_str = f"""
                videotestsrc num-buffers=30 pattern=0 !
                video/x-raw,format=I420,width=640,height=480,framerate=30/1 !
                x264enc bitrate=500 speed-preset=ultrafast tune=zerolatency !
                video/x-h264,profile=baseline !
                h264parse !
                flvmux name=mux !
                rtmpsink location="{rtmp_url}"
            """
            
            logger.info(f"[RTMP_TEST] Testing connection to: {rtmp_url[:50]}...")
            
            # Create and run test pipeline
            test_pipeline = Gst.parse_launch(test_pipeline_str)
            if not test_pipeline:
                logger.error("[RTMP_TEST] Failed to create test pipeline")
                return
            
            # Try to set to PLAYING state
            ret = test_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("[RTMP_TEST] Test pipeline failed to start")
                return
            
            # Wait briefly for connection attempt
            test_bus = test_pipeline.get_bus()
            start_time = time.time()
            
            while time.time() - start_time < 5:  # 5 second test
                msg = test_bus.timed_pop_filtered(100 * Gst.MSECOND, 
                    Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.ELEMENT)
                
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        logger.error(f"[RTMP_TEST] Error: {err}: {debug}")
                        break
                    elif msg.type == Gst.MessageType.WARNING:
                        warn, debug = msg.parse_warning()
                        logger.warning(f"[RTMP_TEST] Warning: {warn}: {debug}")
                    elif msg.type == Gst.MessageType.ELEMENT:
                        struct = msg.get_structure()
                        if struct and "rtmp" in struct.get_name():
                            logger.info(f"[RTMP_TEST] RTMP message: {struct.to_string()}")
            
            # Check final state
            state_ret, state, pending = test_pipeline.get_state(1 * Gst.SECOND)
            if state == Gst.State.PLAYING:
                logger.info("[RTMP_TEST] ✅ RTMP connection appears to be working")
            else:
                logger.error(f"[RTMP_TEST] ❌ RTMP connection failed - State: {state}")
            
            # Cleanup
            test_pipeline.set_state(Gst.State.NULL)
            
        except Exception as e:
            logger.error(f"[RTMP_TEST] Exception during test: {e}")

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[DUAL_STREAMER] GStreamer error: {err}: {debug}")
            # Check if it's a network/RTMP related error
            if "rtmp" in str(err).lower() or "network" in str(err).lower():
                logger.error(f"[DUAL_STREAMER] RTMP/Network error - check stream key and internet connection")
            self.stop_streaming()
        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"[DUAL_STREAMER] GStreamer warning: {err}: {debug}")
        elif message.type == Gst.MessageType.EOS:
            logger.info("[DUAL_STREAMER] End of stream")
            self.stop_streaming()
        elif message.type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == self.pipeline:
                logger.info(f"[DUAL_STREAMER] Pipeline state changed: {old_state.value_nick} -> {new_state.value_nick}")
        elif message.type == Gst.MessageType.STREAM_START:
            logger.info("[DUAL_STREAMER] Stream started")
        elif message.type == Gst.MessageType.APPLICATION:
            logger.info(f"[DUAL_STREAMER] Application message: {message}")
        elif message.type == Gst.MessageType.ELEMENT:
            # Log element messages which might contain RTMP connection info
            struct = message.get_structure()
            if struct:
                logger.debug(f"[DUAL_STREAMER] Element message: {struct.to_string()}")

    def initialize_dual_robot_system(self) -> None:
        """Initialize the dual robot conversation system"""
        logger.info("[DUAL_STREAMER] Initializing dual robot system...")
        
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
            logger.info(f"[DUAL_STREAMER] User message added: {author}: {message}")

    def monitor_youtube_chat(self) -> None:
        """Monitor YouTube chat and forward to robots"""
        logger.info("[DUAL_STREAMER] Starting YouTube chat monitoring...")
        
        while self.is_streaming:
            try:
                messages = self.youtube_api.get_recent_chat_messages()
                
                for message in messages:
                    message_hash = hash(f"{message['author']}{message['message']}{message.get('timestamp', '')}")
                    
                    if message_hash not in self.message_hashes:
                        self.message_hashes.add(message_hash)
                        self.processed_messages.append(message)
                        
                        # Send to robots
                        self.send_user_message_to_robots(message['message'], message['author'])
                        
                        logger.info(f"[DUAL_STREAMER] New chat: {message['author']}: {message['message']}")

                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"[DUAL_STREAMER] YouTube chat error: {e}")
                time.sleep(5)

    def monitor_twitch_chat(self) -> None:
        """Monitor Twitch chat and forward to robots"""
        logger.info("[DUAL_STREAMER] Starting Twitch chat simulation...")
        
        while self.is_streaming:
            try:
                message = self.twitch_chat_sim.get_next_message()
                if message:
                    # Send to robots
                    self.send_user_message_to_robots(message['message'], message['author'])
                    
                    logger.info(f"[DUAL_STREAMER] Simulated chat: {message['author']}: {message['message']}")
                
                time.sleep(random.uniform(5, 15))  # Random intervals
                
            except Exception as e:
                logger.error(f"[DUAL_STREAMER] Twitch chat error: {e}")
                time.sleep(5)

    def start_streaming(self, stream_key: str, platform: str) -> None:
        """Start the dual robot streaming"""
        try:
            self.platform = platform
            self.is_streaming = True
            
            # Setup pipeline
            self.setup_gstreamer_pipeline(stream_key)
            
            # Initialize dual robot system
            self.initialize_dual_robot_system()
            
            # Setup platform-specific chat monitoring
            if platform == "youtube":
                self.youtube_api = YouTubeAPIManager()
                self.youtube_api.initialize()
                threading.Thread(target=self.monitor_youtube_chat, daemon=True).start()
            elif platform == "twitch":
                self.twitch_chat_sim = TwitchChatSimulator(config.TWITCH_SIMULATED_MESSAGES)
                threading.Thread(target=self.monitor_twitch_chat, daemon=True).start()
            
            # Start buffer monitoring (but not waveform updates or audio pushing yet)
            self.buffer_monitor_thread = threading.Thread(target=self.monitor_audio_buffer, daemon=True)
            self.buffer_monitor_thread.start()
            
            # Don't pre-push data - let pipeline negotiate caps first
            logger.info("[DUAL_STREAMER] Skipping initial data push - will start after PLAYING state")
            
            # Start pipeline
            logger.info("[DUAL_STREAMER] Starting GStreamer pipeline...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                # Check bus for immediate error
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    logger.error(f"❌ [PIPELINE_IMMEDIATE_ERROR] {err}: {debug}")
                raise RuntimeError("Failed to start GStreamer pipeline")
            
            # Wait for pipeline to reach PLAYING state with detailed monitoring
            logger.info("[DUAL_STREAMER] Waiting for pipeline to start...")
            
            # Monitor bus messages during state change
            bus = self.pipeline.get_bus()
            timeout = 10  # 10 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check for messages
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND, 
                    Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.STATE_CHANGED)
                
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        logger.error(f"❌ [PIPELINE_ERROR] {err}: {debug}")
                        # Don't return, let state check below handle it
                    elif msg.type == Gst.MessageType.WARNING:
                        warn, debug = msg.parse_warning()
                        logger.warning(f"⚠️ [PIPELINE_WARNING] {warn}: {debug}")
                    elif msg.type == Gst.MessageType.STATE_CHANGED:
                        if msg.src == self.pipeline:
                            old, new, pending = msg.parse_state_changed()
                            logger.info(f"[PIPELINE_STATE] {old.value_nick} -> {new.value_nick}")
                            if new == Gst.State.PLAYING:
                                logger.info("✅ [DUAL_STREAMER] Pipeline reached PLAYING state!")
                                break
                
                # Check current state
                state_ret, state, pending = self.pipeline.get_state(0)  # Non-blocking check
                if state == Gst.State.PLAYING:
                    logger.info("✅ [DUAL_STREAMER] Pipeline is now PLAYING")
                    break
                elif state_ret == Gst.StateChangeReturn.FAILURE:
                    logger.error(f"❌ [DUAL_STREAMER] Pipeline state change failed")
                    break
            
            # Final state check
            state_ret, state, pending = self.pipeline.get_state(1 * Gst.SECOND)
            if state_ret != Gst.StateChangeReturn.SUCCESS or state != Gst.State.PLAYING:
                logger.error(f"❌ [DUAL_STREAMER] Final state check - Return: {state_ret}, State: {state}, Pending: {pending}")
                
                # Get any final error messages
                while True:
                    msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.WARNING)
                    if not msg:
                        break
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        logger.error(f"❌ [FINAL_ERROR] {err}: {debug}")
                    elif msg.type == Gst.MessageType.WARNING:
                        warn, debug = msg.parse_warning()
                        logger.warning(f"⚠️ [FINAL_WARNING] {warn}: {debug}")
                        
                # Try to test RTMP connection separately
                logger.info("[DUAL_STREAMER] Testing RTMP connection...")
                self._test_rtmp_connection(stream_key)
            else:
                logger.info("✅ [DUAL_STREAMER] Pipeline successfully started!")
                
                # Now that pipeline is running, start data processing threads
                logger.info("[DUAL_STREAMER] Starting audio/waveform processing...")
                self.waveform_running = True
                if hasattr(self, 'waveform_update_thread'):
                    self.waveform_update_thread = threading.Thread(target=self._waveform_update_loop, daemon=True)
                    self.waveform_update_thread.start()
            
            logger.info("🎬 Dual robot streaming started!")
            
            # Keep running
            try:
                while self.is_streaming:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Streaming interrupted by user")
            
        except Exception as e:
            logger.error(f"[DUAL_STREAMER] Streaming error: {e}")
            raise
        finally:
            self.stop_streaming()

    def _waveform_update_loop(self):
        """Update waveforms at video framerate"""
        frame_time = 1.0 / config.VIDEO_FRAMERATE
        
        while self.waveform_running and self.is_streaming:
            start_time = time.time()
            
            # The actual waveform update happens in _push_waveform_data
            # This just maintains the timing
            
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

    def monitor_audio_buffer(self):
        """Monitor dual audio buffer health"""
        while self.is_streaming:
            try:
                health = self.audio_mixer.get_buffer_health()
                left_speaking, right_speaking = self.audio_mixer.get_robot_speaking_status()
                
                if health["left_seconds"] > 2.0 or health["right_seconds"] > 2.0:
                    logger.info(
                        f"[DUAL_STREAMER] Buffer status - Left: {health['left_seconds']:.1f}s "
                        f"(speaking: {left_speaking}), Right: {health['right_seconds']:.1f}s "
                        f"(speaking: {right_speaking})"
                    )
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"[DUAL_STREAMER] Buffer monitor error: {e}")
                time.sleep(2)

    def stop_streaming(self) -> None:
        """Stop the dual robot streaming"""
        logger.info("[DUAL_STREAMER] Stopping streaming...")
        
        self.is_streaming = False
        self.waveform_running = False
        
        # Stop dual robot system
        if self.dual_robot_manager:
            self.dual_robot_manager.stop_conversation()
        
        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Clear buffers
        if self.audio_mixer:
            self.audio_mixer.clear_buffers()
        
        if self.waveform_generator:
            self.waveform_generator.clear_waveforms()
        
        logger.info("🛑 Dual robot streaming stopped")

    def run(self, platform: str, broadcast_id: str = None, stream_key: str = None) -> None:
        """Run the dual robot streamer"""
        logger.info(f"[DUAL_STREAMER] Starting dual robot streamer for {platform}")
        
        try:
            if platform == "youtube":
                if stream_key:
                    self.start_streaming(stream_key, platform)
                else:
                    # YouTube streaming with API integration would go here
                    logger.error("YouTube API streaming not implemented yet")
            elif platform == "twitch":
                if not stream_key:
                    logger.error("Stream key required for Twitch")
                    return
                self.start_streaming(stream_key, platform)
            else:
                logger.error(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"[DUAL_STREAMER] Failed to start streaming: {e}")
            raise
