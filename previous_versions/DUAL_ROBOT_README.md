# Dual Robot AI Streamer

This system creates a live stream with two AI robots having conversations using OpenAI's Realtime API.

## Features

🤖 **Two Robot Personalities:**
- **Left Robot (RoboChat-L)**: Curious, optimistic, loves to start conversations
- **Right Robot (RoboChat-R)**: Analytical, wise, provides insights and perspectives

🎵 **Audio & Visual:**
- Separate audio streams for each robot with different voices
- Positioned waveform animations over each robot's face
- Mixed audio output for streaming

💬 **User Interaction:**
- User chat messages are stored and incorporated into robot conversations
- Robots reference user comments periodically
- Works with YouTube and Twitch chat

## Setup

1. **Background Video**: Make sure you have `output_fixed2.mp4` with two robots positioned left and right

2. **Configuration**: The system uses these key settings in `config.py`:
   - `ROBOT_POSITIONS`: Coordinates for waveform placement over robot faces
   - `ROBOT_VOICES`: Different voices for each robot ("alloy", "echo")
   - `ROBOT_CONVERSATION_DELAY`: Time between robot responses (3 seconds)

## Usage

### For Twitch Streaming:
```bash
python dual_robot_main.py --platform twitch --stream-key YOUR_TWITCH_STREAM_KEY
```

### For YouTube Streaming:
```bash
python dual_robot_main.py --platform youtube --stream-key YOUR_YOUTUBE_STREAM_KEY
```

## How It Works

1. **Initialization**: Both robots connect to OpenAI Realtime API with different system messages
2. **Conversation Flow**: 
   - Left robot starts with a greeting
   - Robots take turns responding to each other
   - Every 10 turns, they incorporate recent user messages
3. **Audio Processing**: 
   - Each robot's audio is processed separately
   - Waveforms are positioned over their faces
   - Audio streams are mixed for final output
4. **User Integration**: Chat messages are stored and periodically shared with robots

## Robot Personalities

### Left Robot (RoboChat-L)
- Enthusiastic and curious
- Asks questions and starts new topics
- References human interactions
- Warm and optimistic personality

### Right Robot (RoboChat-R)
- Thoughtful and analytical
- Provides insights and connections
- Synthesizes ideas
- Calm and reflective personality

## File Structure

- `dual_robot_main.py` - Main entry point for dual robot system
- `modules/api/dual_robot_realtime.py` - Manages robot conversations
- `modules/audio/dual_mixer.py` - Handles separate audio streams
- `modules/audio/dual_waveform.py` - Generates positioned waveforms
- `modules/streaming/dual_robot_streamer.py` - Main streaming orchestrator

## User Message Storage

User messages are automatically saved to `user_messages.json` and include:
- Author name
- Message content  
- Timestamp
- Automatic cleanup (keeps last 50 messages)

## Troubleshooting

1. **No Audio**: Check that both robots are connecting to OpenAI API
2. **No Waveforms**: Verify robot positions in config match your video
3. **Robots Not Talking**: Check conversation delay and response completion callbacks
4. **Stream Issues**: Ensure background video file exists and GStreamer is properly installed

## Customization

You can customize:
- Robot personalities by editing system messages in `config.py`
- Waveform positions via `ROBOT_POSITIONS`
- Conversation timing with `ROBOT_CONVERSATION_DELAY`
- How often user messages are incorporated with `ROBOT_MAX_CONVERSATION_TURNS` 