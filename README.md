# 🤖 Dual API Gendered Robots - AI Live Streaming System

A real-time AI streaming system featuring two AI robots with distinct personalities and voices that engage in natural conversations while streaming to YouTube. The system uses OpenAI's Realtime API and Google's Gemini Live API to create an interactive, dual-personality streaming experience.

## 🎭 What It Does

- **Two AI Personalities**: 
  - **Left Robot** (Male voice): Uses OpenAI's Realtime API with "echo" voice - curious and enthusiastic
  - **Right Robot** (Female voice): Uses Google's Gemini Live API with "Puck" voice - analytical and thoughtful
- **Live Streaming**: Automatically creates and manages YouTube Live broadcasts
- **Interactive Chat**: Monitors YouTube chat and prioritizes viewer messages over robot conversations
- **Visual Feedback**: Real-time audio waveforms positioned for each speaking robot

## 📋 Prerequisites

### 1. System Requirements
- **Python 3.8+**
- **Windows/Linux/macOS**
- **Stable internet connection** (5+ Mbps upload for streaming)

### 2. Required Software

#### Install Python Dependencies
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install websockets asyncio numpy
pip install PyGObject  # For GStreamer Python bindings
```

#### Install GStreamer
GStreamer is required for video/audio processing:

**Windows:**
1. Download from [GStreamer website](https://gstreamer.freedesktop.org/download/)
2. Install both runtime and development packages
3. Add GStreamer to your PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

**macOS:**
```bash
brew install gstreamer gst-plugins-base gst-plugins-good \
    gst-plugins-bad gst-plugins-ugly gst-libav
```

### 3. Background Video
Place a video file named `output.mp4` in the project root directory. This will be used as the background for your stream.

**Download Test Video**: You can download a ready-to-use test video from [this Google Drive link](https://drive.google.com/file/d/1joxv6PDkry3O-4YCoYImMw8rjdTRFL4b/view?usp=sharing). Save it as `output.mp4` in your project root directory.

## 🔑 API Keys Setup

### 1. OpenAI API Key (Required)
Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Important**: You need access to the **Realtime API** which may require:
- Being on a paid plan
- Having realtime API access enabled on your account

### 2. Google Gemini API Key (Required)
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

1. Click "Get API key"
2. Create a new API key
3. Copy the key

### 3. YouTube API Credentials (Required)

#### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Note your project name

#### Step 2: Enable YouTube Data API v3
1. In your project, go to **APIs & Services** → **Library**
2. Search for "YouTube Data API v3"
3. Click on it and press **Enable**

#### Step 3: Create OAuth 2.0 Credentials
1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth 2.0 Client IDs**
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in required fields (app name, email)
   - Add your email to test users
4. For Application type, select **Desktop app**
5. Name it (e.g., "YouTube Streaming Bot")
6. Click **Create**
7. Download the credentials or copy the **Client ID** and **Client Secret**

## ⚙️ Configuration

### 1. Update modules/config.py

Open `modules/config.py` and add your credentials:

```python
# YouTube Credentials
YOUTUBE_CLIENT_ID = "your_client_id_here"
YOUTUBE_CLIENT_SECRET = "your_client_secret_here"
YOUTUBE_REFRESH_TOKEN = ""  # Leave empty for now

# OpenAI API Key
OPENAI_API_KEY = "sk-your-openai-api-key-here"

# Google Gemini API Key
GEMINI_API_KEY = "your-gemini-api-key-here"
```

### 2. Generate YouTube Refresh Token

After adding your YouTube Client ID and Secret, run:

```bash
cd previous_versions
python get_youtube_token.py
```

**Note**: You need to update `get_youtube_token.py` first:
1. Open `previous_versions/get_youtube_token.py`
2. Add your CLIENT_ID and CLIENT_SECRET directly in the file:
```python
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
```

3. Run the script - it will:
   - Open a browser window
   - Ask you to log in to your YouTube account
   - Request permissions for streaming
   - Display a refresh token

4. Copy the refresh token and add it to `modules/config.py`:
```python
YOUTUBE_REFRESH_TOKEN = "your_refresh_token_here"
```

## 🚀 Running the System

### Basic YouTube Streaming
```bash
python dual_api_gendered_robots.py --platform youtube
```

### With Existing YouTube Broadcast
If you already have a broadcast created:
```bash
python dual_api_gendered_robots.py --platform youtube --broadcast-id YOUR_BROADCAST_ID
```

### Alternative: Simple Runner Script
```bash
python previous_versions/run_youtube_robots.py
```
This will prompt you for options and run the appropriate command.

## 🎮 What Happens When You Run It

1. **Authentication**: Connects to YouTube, OpenAI, and Gemini APIs
2. **Broadcast Creation**: Creates a new YouTube Live broadcast (or uses existing)
3. **Stream Setup**: Configures the video/audio pipeline
4. **Robot Initialization**: Both AI robots connect and prepare for conversation
5. **Go Live**: Stream transitions to live status on YouTube
6. **Conversation Starts**: Robots begin talking to each other
7. **Chat Monitoring**: System watches for YouTube chat messages
8. **Interactive Responses**: When viewers chat, robots respond directly to them

## 💬 How Chat Integration Works

- Viewer messages in YouTube chat are captured in real-time
- Messages interrupt robot-to-robot conversation
- The active robot responds directly to the viewer by name
- After responding, robots resume their conversation
- All messages are prioritized: User chat > Robot conversation

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. "invalid_grant: Token has been expired or revoked"
- Your YouTube refresh token has expired
- Run `get_youtube_token.py` again to generate a new one
- Update the new token in `config.py`

#### 2. "No module named 'gi'"
- GStreamer Python bindings not installed
- Windows: Ensure you installed GStreamer development package
- Linux: `sudo apt-get install python3-gi`
- macOS: `brew install pygobject3`

#### 3. "GStreamer plugin not found"
- Missing GStreamer plugins
- Install the full GStreamer suite (base, good, bad, ugly plugins)

#### 4. "API key not valid"
- Double-check your API keys in `config.py`
- Ensure no extra spaces or quotes
- Verify APIs are enabled in respective consoles

#### 5. Stream not appearing on YouTube
- Wait 10-30 seconds after "transitioned to live"
- Check YouTube Studio for the live stream
- Ensure your account can live stream (may need verification)

#### 6. No audio from robots
- Check OpenAI Realtime API access
- Verify Gemini API key has necessary permissions
- Monitor logs for API connection errors

### Debug Mode
To see detailed logs:
```python
# In dual_api_gendered_robots.py, change:
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Optimization

### Recommended Settings
- **Internet**: Minimum 5 Mbps upload speed
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: SSD for video file

### Adjustable Parameters in config.py
```python
# Reduce delays for faster conversations
ROBOT_MIN_PAUSE = 0.05  # Minimum pause between robots
ROBOT_MAX_PAUSE = 0.15  # Maximum pause between robots

# Video quality (lower if connection is poor)
VIDEO_BITRATE = 2500  # Can reduce to 1500 for lower bandwidth

# Audio settings
AUDIO_BITRATE = 128000  # Can reduce to 96000 if needed
```

## 🔒 Security Notes

- **Never commit API keys** to version control
- Keep your `config.py` file private
- Use environment variables for production deployments
- Regularly rotate your API keys
- Monitor API usage to avoid unexpected charges

## 📝 Additional Notes

- The system requires continuous internet connection
- YouTube may have daily streaming quotas
- API costs apply for OpenAI and Google services
- Test with unlisted broadcasts before going public
- Monitor chat for inappropriate content

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the full logs for error messages
3. Ensure all prerequisites are properly installed
4. Verify API quotas and limits haven't been exceeded
5. Check that your YouTube account can live stream

## 🎯 Quick Checklist

Before running, ensure you have:
- [ ] Python 3.8+ installed
- [ ] GStreamer installed and in PATH
- [ ] All Python dependencies installed
- [ ] `output.mp4` video file in project root
- [ ] OpenAI API key with Realtime API access
- [ ] Google Gemini API key
- [ ] YouTube OAuth credentials (Client ID & Secret)
- [ ] YouTube refresh token generated
- [ ] All keys added to `modules/config.py`
- [ ] Stable internet connection

Once everything is set up, you're ready to start streaming with your AI robots! 