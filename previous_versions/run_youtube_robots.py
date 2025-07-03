#!/usr/bin/env python3
"""
Start Dual-API Robot Streamer for YouTube with Chat Integration
"""

import sys
import subprocess

def main():
    print("🤖📺 Starting YouTube Robot Stream with Chat Integration...")
    print()
    print("This will:")
    print("✅ Stream to YouTube") 
    print("✅ Enable YouTube chat monitoring")
    print("✅ Route chat messages to robots for direct responses")
    print("✅ Maintain ultra-fast robot conversations")
    print("✅ Use both OpenAI (male) and Gemini (female) APIs")
    print()
    
    # You can provide a broadcast ID if you have an existing stream
    broadcast_id = input("Enter existing YouTube broadcast ID (or press Enter to create new): ").strip()
    
    cmd = [sys.executable, "dual_api_gendered_robots.py", "--platform", "youtube"]
    
    if broadcast_id:
        cmd.extend(["--broadcast-id", broadcast_id])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Stream stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Stream failed with exit code {e.returncode}")

if __name__ == "__main__":
    main() 