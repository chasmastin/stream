#!/usr/bin/env python3
"""
Stream Performance Monitor
Analyzes log output to track stream health
"""

import re
import sys
from collections import defaultdict, deque
from datetime import datetime


class StreamMonitor:
    def __init__(self):
        self.buffer_states = deque(maxlen=100)
        self.push_rates = deque(maxlen=50)
        self.warnings = defaultdict(int)
        self.positioning_calls = 0
        self.last_robot = None
        
    def parse_log_line(self, line):
        """Parse a single log line for relevant metrics"""
        
        # Track positioning calls and robot switches
        if "positioning waveform" in line.lower():
            self.positioning_calls += 1
            if "LEFT robot" in line:
                current_robot = "left"
            elif "RIGHT robot" in line:
                current_robot = "right"
            else:
                current_robot = None
                
            if current_robot != self.last_robot:
                print(f"✅ Robot switch: {self.last_robot} → {current_robot}")
                self.last_robot = current_robot
        
        # Track audio buffer states
        buffer_match = re.search(r'Mixer: (\d+\.\d+)s \((\w+)\)', line)
        if buffer_match:
            buffer_time = float(buffer_match.group(1))
            buffer_state = buffer_match.group(2)
            self.buffer_states.append((buffer_time, buffer_state))
            
        # Track push rates
        rate_match = re.search(r'Rate: (\d+) pushes/sec', line)
        if rate_match:
            push_rate = int(rate_match.group(1))
            self.push_rates.append(push_rate)
            
        # Track warnings
        if "WARNING" in line:
            if "silence" in line.lower():
                self.warnings["silence"] += 1
            elif "buffer" in line.lower():
                self.warnings["buffer"] += 1
            else:
                self.warnings["other"] += 1
                
        # Track burst mode
        if "BURST DETECTED" in line:
            burst_match = re.search(r'(\d+\.\d+)x incoming rate', line)
            if burst_match:
                rate = float(burst_match.group(1))
                print(f"🚀 BURST: {rate}x incoming rate")
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("STREAM PERFORMANCE SUMMARY")
        print("="*60)
        
        # Buffer health
        if self.buffer_states:
            recent_buffers = list(self.buffer_states)[-10:]
            avg_buffer = sum(b[0] for b in recent_buffers) / len(recent_buffers)
            states = [b[1] for b in recent_buffers]
            print(f"📊 Buffer Average: {avg_buffer:.2f}s")
            print(f"📊 Recent States: {', '.join(states[-5:])}")
        
        # Push rates
        if self.push_rates:
            avg_rate = sum(self.push_rates) / len(self.push_rates)
            print(f"📊 Avg Push Rate: {avg_rate:.1f} pushes/sec")
        
        # Warnings summary
        print(f"⚠️  Warnings: {dict(self.warnings)}")
        print(f"🔄 Positioning Calls: {self.positioning_calls}")
        
        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        if self.warnings["silence"] > 5:
            print("- High silence warnings: Consider increasing buffer size")
        if avg_rate < 45:
            print("- Low push rate: Check for CPU/network bottlenecks")
        if self.positioning_calls > 20:
            print("- Frequent positioning: Check robot switching logic")
        
        print("="*60)


def main():
    monitor = StreamMonitor()
    
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            for line in f:
                monitor.parse_log_line(line.strip())
    else:
        # Read from stdin (pipe)
        print("📡 Stream Monitor Active - Reading log lines...")
        print("Usage: python stream_monitor.py [logfile] or pipe logs to stdin")
        print("-" * 60)
        
        try:
            for line in sys.stdin:
                monitor.parse_log_line(line.strip())
        except KeyboardInterrupt:
            pass
    
    monitor.print_summary()


if __name__ == "__main__":
    main() 