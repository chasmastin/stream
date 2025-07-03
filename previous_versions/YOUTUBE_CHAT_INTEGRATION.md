# 🤖📺 YouTube Chat Integration with Dual-API Robots

## Overview
The dual-API robot system now has **full YouTube chat integration** where viewers can interact directly with both AI robots (OpenAI + Gemini) in real-time.

## How It Works

### 🔄 **Chat Flow**
1. **Viewer posts message** in YouTube live chat
2. **System captures message** with adaptive polling (500ms-2s intervals)
3. **Message routed to active robot** (or defaults to left/OpenAI robot)
4. **Robot responds directly** to the viewer by name
5. **Response includes chat context** for natural conversation

### 🎭 **Robot Behavior**

#### **Left Robot (OpenAI - Male Voice)**
- **Personality**: Curious and enthusiastic
- **Chat Response**: "Hey [username]! That's awesome - [enthusiastic response]!"
- **Voice**: Echo (male)
- **Max Length**: 15 words

#### **Right Robot (Gemini - Female Voice)**  
- **Personality**: Wise and analytical
- **Chat Response**: "Great question [username]! [thoughtful analysis]"
- **Voice**: Puck (female)
- **Max Length**: 15 words

### ⚡ **Performance Features**

- **Ultra-Fast Polling**: 500ms minimum, respects YouTube's rate limits
- **Smart Routing**: Messages go to currently active robot
- **Interrupt Capability**: Chat messages can interrupt ongoing robot conversations
- **Auto-Cleanup**: Message history managed automatically (max 1000 messages)
- **Rate Limit Protection**: 60-second backoff on quota exceeded

## 🚀 **Starting YouTube Stream with Chat**

### Option 1: Simple Start
```bash
python run_youtube_robots.py
```

### Option 2: Direct Command
```bash
python dual_api_gendered_robots.py --platform youtube
```

### Option 3: With Existing Broadcast
```bash
python dual_api_gendered_robots.py --platform youtube --broadcast-id YOUR_BROADCAST_ID
```

## 💬 **Chat Integration Features**

### **Adaptive Polling**
- Starts at 500ms intervals
- Speeds up during active chat (250ms)
- Slows down when quiet (2s)
- Respects YouTube's `pollingIntervalMillis` suggestions

### **Message Processing**
- Deduplication using message hashes
- Real-time text overlay updates
- Direct viewer name mentions
- Context-aware responses

### **Error Handling**
- Quota exceeded protection
- Forbidden/ban detection
- Graceful error recovery
- Detailed logging

## 🔧 **Configuration**

### **Robot Instructions Enhanced**
Both robots now prioritize chat responses:
- **OpenAI**: "When viewers send chat messages, prioritize responding to them directly with enthusiasm!"
- **Gemini**: "When viewers send chat messages, prioritize responding to them directly with wisdom and analysis!"

### **Chat Prompt Format**
```
A viewer named [username] in the chat just said: '[message]' - Please respond to them directly and acknowledge their message!
```

## 📊 **Logging**

Chat activity is logged with prefixes:
- `💬📺 YouTube Chat - [username]: [message]` 
- `📤💬 Sending chat message to RIGHT robot (Gemini): [username]`
- `📤💬 Sending chat message to LEFT robot (OpenAI): [username]`

## ⚙️ **Technical Details**

### **Chat Monitoring Thread**
- Runs in background daemon thread
- Uses `youtube_api.get_chat_messages_fast()`
- Maintains `next_page_token` for continuity
- Auto-cleanup of message hashes

### **Integration Points**
- `monitor_youtube_chat()` - Main chat polling loop
- `update_text_overlay()` - Visual chat display
- `send_text_message()` - Route to appropriate robot API
- Message routing based on `current_speaking_robot`

## 🎯 **Expected Behavior**

1. **Stream starts** → Robots begin conversation
2. **Viewer chats** → Active robot immediately responds
3. **Multiple viewers** → Robots alternate responses
4. **Quiet periods** → Robots resume their conversation
5. **High activity** → Robots prioritize chat over their dialogue

The system creates a natural flow between **robot-to-robot conversations** and **robot-to-viewer interactions**, making the stream highly engaging and interactive! 