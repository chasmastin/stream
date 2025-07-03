"""
Twitch Chat Simulator for testing
"""

import time
import random
import logging
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class TwitchChatSimulator:
    """Simulates Twitch chat messages"""

    def __init__(self, messages: List[Dict[str, str]]):
        self.message_pool = messages.copy()
        self.used_messages = []
        self.last_message_time = time.time()
        self.min_interval = 30
        self.max_interval = 60

    def get_next_message(self) -> Optional[Dict[str, str]]:
        """Get next simulated message"""
        current_time = time.time()
        time_since_last = current_time - self.last_message_time

        next_interval = random.uniform(self.min_interval, self.max_interval)

        if time_since_last >= next_interval:
            if not self.message_pool:
                self.message_pool = self.used_messages.copy()
                self.used_messages = []
                random.shuffle(self.message_pool)

            if self.message_pool:
                message = self.message_pool.pop(
                    random.randint(0, len(self.message_pool) - 1)
                )
                self.used_messages.append(message)
                self.last_message_time = current_time
                message["timestamp"] = datetime.now().isoformat()
                return message

        return None 