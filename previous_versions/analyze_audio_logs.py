#!/usr/bin/env python3
"""
Script to analyze audio logs from the streaming application.
Usage: python analyze_audio_logs.py < logfile.txt
"""

import sys
import re
from collections import defaultdict

def analyze_logs():
    """Analyze audio logs from stdin"""
    
    # Counters
    stats = defaultdict(int)
    buffer_levels = []
    push_failures = []
    warnings = []
    incoming_rates = []
    push_modes = defaultdict(int)
    
    print("=== Audio Log Analysis ===\n")
    
    for line in sys.stdin:
        # Extract audio-related logs
        if "[AUDIO]" in line or "[MIXER]" in line or "[OPENAI]" in line:
            # Print the line
            print(line.strip())
            
            # Analyze buffer levels
            if "Mixer buffer:" in line and "Push stats" not in line:
                match = re.search(r"Mixer buffer: ([\d.]+)s", line)
                if match:
                    buffer_levels.append(float(match.group(1)))
            
            # Analyze incoming rate
            if "Incoming rate:" in line:
                match = re.search(r"= ([\d.]+)x realtime", line)
                if match:
                    incoming_rates.append(float(match.group(1)))
            
            # Analyze push modes
            if "Mode:" in line:
                mode_match = re.search(r"Mode: (\w+)", line)
                if mode_match:
                    mode = mode_match.group(1)
                    push_modes[mode] += 1
            
            # Track target speeds
            if "target:" in line:
                speed_match = re.search(r"target: ([\d.]+)x", line)
                if speed_match:
                    speed = float(speed_match.group(1))
                    if speed > 1.2:
                        stats["high_speed_periods"] += 1
            
            # Count events
            if "[OPENAI] Audio chunk:" in line:
                stats["openai_chunks"] += 1
            elif "[MIXER] Received" in line and "bytes of audio" in line:
                stats["mixer_received"] += 1
            elif "[AUDIO] Pushed" in line:
                stats["audio_pushed"] += 1
            elif "Push failed" in line:
                push_failures.append(line.strip())
                stats["push_failures"] += 1
            elif "WARNING" in line or "warning" in line:
                warnings.append(line.strip())
                stats["warnings"] += 1
            elif "Buffer critical" in line:
                if "shrinking" in line:
                    stats["buffer_critical_shrinking"] += 1
                else:
                    stats["buffer_critical"] += 1
            elif "Audio stream started" in line:
                stats["stream_starts"] += 1
            elif "Audio response completed" in line:
                stats["stream_completes"] += 1
            elif "catch-up mode" in line or "Emergency buffer clear" in line:
                stats["catch_up_mode_switches"] += 1
    
    # Print summary
    print("\n=== Summary ===")
    print(f"OpenAI audio chunks received: {stats['openai_chunks']}")
    print(f"Mixer chunks received: {stats['mixer_received']}")
    print(f"Audio chunks pushed to GStreamer: {stats['audio_pushed']}")
    print(f"Push failures: {stats['push_failures']}")
    print(f"Warnings: {stats['warnings']}")
    print(f"Buffer critical events: {stats['buffer_critical']} (shrinking: {stats['buffer_critical_shrinking']})")
    print(f"Audio streams: {stats['stream_starts']} started, {stats['stream_completes']} completed")
    print(f"Speed adjustments: {stats['catch_up_mode_switches']}")
    print(f"High speed periods (>1.2x): {stats['high_speed_periods']}")
    
    if incoming_rates:
        print(f"\nIncoming audio rates - Min: {min(incoming_rates):.1f}x, Max: {max(incoming_rates):.1f}x, Avg: {sum(incoming_rates)/len(incoming_rates):.1f}x realtime")
    
    if push_modes:
        total_pushes = sum(push_modes.values())
        print(f"\nPush modes breakdown:")
        for mode in ['normal', 'slight', 'moderate', 'high', 'emergency']:
            count = push_modes.get(mode, 0)
            if count > 0:
                print(f"  {mode.capitalize()}: {count} ({count/total_pushes*100:.1f}%)")
    
    if buffer_levels:
        print(f"\nBuffer levels - Min: {min(buffer_levels):.2f}s, Max: {max(buffer_levels):.2f}s, Avg: {sum(buffer_levels)/len(buffer_levels):.2f}s")
    
    if push_failures:
        print(f"\n=== Push Failures ({len(push_failures)}) ===")
        for failure in push_failures[:5]:  # Show first 5
            print(failure)
        if len(push_failures) > 5:
            print(f"... and {len(push_failures) - 5} more")
    
    if warnings:
        print(f"\n=== Warnings ({len(warnings)}) ===")
        # Group similar warnings
        warning_counts = defaultdict(int)
        for warning in warnings:
            if "Adding" in warning and "silence during speech" in warning:
                warning_counts["Adding silence during speech"] += 1
            else:
                warning_counts[warning] += 1
        
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count > 1:
                print(f"{warning_type} (x{count})")
            else:
                print(warning_type)

if __name__ == "__main__":
    analyze_logs() 