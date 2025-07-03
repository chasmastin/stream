"""
Test script to validate Gemini Live API transcription capabilities
"""
import asyncio
import json
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gemini_transcription():
    """Test if Gemini Live API supports transcription"""
    
    # You'll need to set your API key
    api_key = "YOUR_GEMINI_API_KEY"  # Replace with actual key
    
    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
    
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            logger.info("✅ Connected to Gemini Live API")
            
            # Setup message with transcription enabled
            setup_message = {
                "setup": {
                    "model": "models/gemini-2.0-flash-live-001",
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "temperature": 0.7,
                        "max_output_tokens": 2048
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": "You are a helpful assistant. Keep responses brief."
                        }]
                    },
                    "tools": [],
                    # THE KEY TRANSCRIPTION SETTINGS
                    "input_audio_transcription": {},
                    "output_audio_transcription": {}
                }
            }
            
            await ws.send(json.dumps(setup_message))
            logger.info("📤 Setup message sent with transcription enabled")
            
            # Send a simple text message to trigger response
            client_content = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": "Say hello in a friendly voice"}]
                    }],
                    "turnComplete": True
                }
            }
            
            await ws.send(json.dumps(client_content))
            logger.info("📤 Test message sent")
            
            # Listen for responses
            response_count = 0
            async for message in ws:
                if response_count > 10:  # Limit responses
                    break
                    
                data = json.loads(message)
                response_count += 1
                
                if "setupComplete" in data:
                    logger.info("✅ Setup completed")
                    continue
                    
                if "serverContent" in data:
                    server_content = data["serverContent"]
                    
                    # Check for transcription
                    if "outputTranscription" in server_content:
                        transcription = server_content["outputTranscription"]
                        logger.info(f"🎯 OUTPUT TRANSCRIPTION FOUND: {transcription}")
                        
                    if "inputTranscription" in server_content:
                        transcription = server_content["inputTranscription"] 
                        logger.info(f"🎯 INPUT TRANSCRIPTION FOUND: {transcription}")
                        
                    if server_content.get("turnComplete"):
                        logger.info("✅ Turn completed")
                        break
                        
                logger.debug(f"Raw response: {json.dumps(data, indent=2)}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Gemini Live API Transcription...")
    print("⚠️  Make sure to set your API key in the script!")
    print("📝 This will test if transcription features are available")
    
    # Uncomment to run the test (after setting API key)
    # asyncio.run(test_gemini_transcription())
    
    print("✨ Test script ready - set API key and uncomment the test call") 