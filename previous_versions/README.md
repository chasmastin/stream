# AI Streamer - Modular Architecture

This project has been refactored into a modular architecture for better maintainability and to work better with AI context windows.

## Directory Structure

```
.
├── main.py                    # Main entry point
├── modules/                   # All modules
│   ├── __init__.py
│   ├── config.py             # All configuration constants
│   ├── audio/                # Audio processing modules
│   │   ├── __init__.py
│   │   ├── mixer.py          # SimpleAudioMixer class
│   │   └── waveform.py       # AudioWaveformGenerator class
│   ├── api/                  # API client modules
│   │   ├── __init__.py
│   │   └── openai_realtime.py # OpenAI Realtime API client
│   ├── platforms/            # Platform-specific modules
│   │   ├── __init__.py
│   │   ├── youtube.py        # YouTube API manager
│   │   └── twitch.py         # Twitch chat simulator
│   └── streaming/            # Streaming related modules
│       ├── __init__.py
│       ├── pipeline.py       # GStreamer pipeline builder
│       └── streamer.py       # Main streamer class
└── stream.py                 # Original monolithic file (kept for reference)
```

## Module Descriptions

### `modules/config.py`
Contains all configuration constants including:
- API keys and credentials
- Video/audio settings
- Waveform visualization settings
- Platform-specific URLs
- AI personality instructions

### `modules/audio/`
Audio processing components:
- **`mixer.py`**: Handles audio buffering and mixing with noise reduction
- **`waveform.py`**: Generates visual waveform from audio data

### `modules/api/`
External API clients:
- **`openai_realtime.py`**: WebSocket client for OpenAI's Realtime API

### `modules/platforms/`
Platform-specific implementations:
- **`youtube.py`**: YouTube Live API management (broadcasts, streams, chat)
- **`twitch.py`**: Twitch chat simulation for testing

### `modules/streaming/`
Core streaming functionality:
- **`pipeline.py`**: GStreamer pipeline construction helpers
- **`streamer.py`**: Main streamer class that orchestrates everything

## Usage

The usage remains the same as before:

```bash
# Stream to YouTube
python main.py --platform youtube

# Stream to Twitch
python main.py --platform twitch --stream-key YOUR_STREAM_KEY

# Use existing YouTube broadcast
python main.py --platform youtube --broadcast-id BROADCAST_ID
```

## Benefits of Modular Structure

1. **Better AI Context Management**: Each module is focused and smaller, making it easier to work with AI assistants that have limited context windows.

2. **Easier Maintenance**: Changes to specific functionality (e.g., audio processing) can be made without touching other parts.

3. **Improved Testing**: Individual modules can be tested in isolation.

4. **Code Reusability**: Modules can be imported and used in other projects.

5. **Clearer Organization**: It's easier to understand what each part does and where to find specific functionality.

## Working with Modules

When you need to modify specific functionality:

- **Audio issues?** → Look in `modules/audio/`
- **API changes?** → Check `modules/api/`
- **Platform-specific?** → See `modules/platforms/`
- **Pipeline/GStreamer?** → Check `modules/streaming/`
- **Configuration?** → Edit `modules/config.py`

Each module is designed to be self-contained with clear interfaces, making it easier to understand and modify.

## Audio Debugging

The application includes comprehensive logging for audio debugging. Audio-related logs are prefixed with:
- `[OPENAI]` - Audio received from OpenAI Realtime API
- `[MIXER]` - Audio processing in the mixer
- `[AUDIO]` - Audio pushing to GStreamer

### Capturing Logs

To capture logs for debugging:
```bash
python main.py 2>&1 | tee stream_logs.txt
```

### Analyzing Audio Logs

Use the included log analyzer to extract and summarize audio issues:
```bash
python analyze_audio_logs.py < stream_logs.txt
```

This will show:
- Audio flow statistics
- Buffer levels over time
- Push failures and warnings
- Summary of audio events

### Common Audio Issues

1. **Buffer Growing Too Large**: If you see "Buffer very high" messages, audio is arriving faster than being consumed
2. **Audio Glitches**: Check for "Adding silence during speech" warnings indicating buffer underruns
3. **Push Failures**: Look for "Push failed" messages indicating GStreamer pipeline issues

### Quality-Focused Audio System

The application uses a quality-focused speed adjustment system that prioritizes natural-sounding audio over aggressive buffer management. This handles OpenAI's bursty audio delivery (which can range from 0.1x to 5.5x realtime) while maintaining high audio quality:

**Speed Tiers** (Conservative limits for quality):
- **Normal Mode** (buffer 0.3-1.5s): Real-time playback (1.0x)
- **Slight Mode** (buffer 1.5-4s): Barely noticeable speedup (1.02x) 
- **Moderate Mode** (buffer 4-8s): Very gentle catch-up (1.1x)
- **High Mode** (buffer 8-12s): Gentle catch-up (1.2x)
- **Very High Mode** (buffer 12-20s): Moderate catch-up (1.35x max)
- **Emergency Mode** (buffer > 20s): Automatic buffer clearing

**Key Features**:
- **Quality First**: Maximum 1.35x speed to maintain natural voice
- **Gradual Transitions**: Speed changes at 0.01 per cycle (very slow)
- **20s Buffer Tolerance**: Allows up to 20s buffer before emergency clear
- **Burst Awareness**: Detects OpenAI bursts but doesn't overreact
- **Natural Sound**: Prioritizes audio quality over real-time accuracy

**Why This Approach?**
- OpenAI delivers audio in extreme bursts (up to 5.5x realtime)
- Aggressive speed matching (1.5x+) makes voices sound unnatural
- Allowing larger buffers with gentle catch-up preserves quality
- Users prefer slightly delayed but natural-sounding audio

**Tuning Tips**:
- If audio sounds rushed: Reduce max speeds further
- If buffer grows too large: Increase speed limits slightly
- Emergency clear at 20s prevents excessive memory usage
- Monitor logs for "⚠️ HIGH SPEED" warnings

## Requirements

// ... existing code ... 