"""
Platform-specific modules
"""

from .youtube import YouTubeAPIManager
from .twitch import TwitchChatSimulator

__all__ = ['YouTubeAPIManager', 'TwitchChatSimulator'] 