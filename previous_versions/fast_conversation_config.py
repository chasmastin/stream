#!/usr/bin/env python3
"""
Fast Conversation Configuration Tool
Easily adjust delays between robot conversations
"""

import re
import sys

def update_config_value(config_file, variable_name, new_value):
    """Update a configuration value in the config file"""
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Pattern to match the variable assignment
        pattern = rf'^{variable_name}\s*=\s*[0-9.]+.*$'
        replacement = f'{variable_name} = {new_value}  # Updated for faster conversations'
        
        # Update the value
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(config_file, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {variable_name} = {new_value}")
        return True
    except Exception as e:
        print(f"❌ Error updating {variable_name}: {e}")
        return False

def show_current_config():
    """Show current conversation timing configuration"""
    try:
        with open('modules/config.py', 'r') as f:
            content = f.read()
        
        print("📊 CURRENT CONVERSATION TIMING:")
        print("-" * 40)
        
        # Extract current values
        for var in ['ROBOT_MIN_PAUSE', 'ROBOT_MAX_PAUSE', 'SPEECH_END_DELAY']:
            pattern = rf'^{var}\s*=\s*([0-9.]+)'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                print(f"{var:<20} = {value}s")
        
        print("-" * 40)
    except Exception as e:
        print(f"❌ Error reading config: {e}")

def main():
    if len(sys.argv) == 1:
        # Show current configuration
        show_current_config()
        print("\n🎚️  FAST CONVERSATION PRESETS:")
        print("python3 fast_conversation_config.py ultra-fast    # 0.5-1.5s delays")
        print("python3 fast_conversation_config.py fast         # 1.0-2.0s delays")
        print("python3 fast_conversation_config.py normal       # 1.5-3.0s delays")
        print("python3 fast_conversation_config.py custom MIN MAX SPEECH_DELAY")
        return
    
    preset = sys.argv[1].lower()
    
    if preset == "ultra-fast":
        min_pause = 0.5
        max_pause = 1.5
        speech_delay = 0.1
        print("🚀 Setting ULTRA-FAST conversation timing...")
    elif preset == "fast":
        min_pause = 1.0
        max_pause = 2.0
        speech_delay = 0.2
        print("⚡ Setting FAST conversation timing...")
    elif preset == "normal":
        min_pause = 1.5
        max_pause = 3.0
        speech_delay = 0.3
        print("📢 Setting NORMAL conversation timing...")
    elif preset == "custom":
        if len(sys.argv) != 5:
            print("❌ Custom usage: python3 fast_conversation_config.py custom MIN MAX SPEECH_DELAY")
            return
        try:
            min_pause = float(sys.argv[2])
            max_pause = float(sys.argv[3])
            speech_delay = float(sys.argv[4])
            print(f"🎛️  Setting CUSTOM conversation timing...")
        except ValueError:
            print("❌ Error: All values must be numbers")
            return
    else:
        print("❌ Unknown preset. Use: ultra-fast, fast, normal, or custom")
        return
    
    # Update configuration
    config_file = 'modules/config.py'
    success = True
    
    success &= update_config_value(config_file, 'ROBOT_MIN_PAUSE', min_pause)
    success &= update_config_value(config_file, 'ROBOT_MAX_PAUSE', max_pause)
    success &= update_config_value(config_file, 'SPEECH_END_DELAY', speech_delay)
    
    if success:
        print(f"\n✅ Configuration updated successfully!")
        print(f"Min pause: {min_pause}s")
        print(f"Max pause: {max_pause}s") 
        print(f"Speech end delay: {speech_delay}s")
        print("\n🔄 Restart your stream to apply the new settings.")
    else:
        print("\n❌ Some configuration updates failed.")

if __name__ == "__main__":
    main() 