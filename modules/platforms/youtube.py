"""
YouTube API Manager for live streaming
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class YouTubeAPIManager:
    """YouTube API manager with ultra-fast chat fetching"""

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, scopes: list):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.scopes = scopes
        
        self.youtube = None
        self.broadcast_id = None
        self.stream_id = None
        self.live_chat_id = None
        self.stream_key = None
        self._chat_request = None  # Reusable request object

    def authenticate(self) -> None:
        """Authenticate with YouTube"""
        try:
            creds = Credentials(
                None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=self.scopes,
            )

            creds.refresh(Request())
            self.youtube = build("youtube", "v3", credentials=creds)
            logger.info("✅ YouTube authenticated")

        except Exception as e:
            logger.error(f"YouTube auth error: {e}")
            raise

    def create_broadcast(self, title: str = None, description: str = None) -> str:
        """Create YouTube broadcast"""
        try:
            if not title:
                title = f"🤖 AI Stream - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if not description:
                description = "AI stream with ultra-low latency voice responses and smooth audio waveform visualization!"

            broadcast_response = (
                self.youtube.liveBroadcasts()
                .insert(
                    part="snippet,status,contentDetails",
                    body={
                        "snippet": {
                            "title": title,
                            "description": description,
                            "scheduledStartTime": datetime.utcnow().isoformat() + "Z",
                            "enableLiveChat": True,
                        },
                        "status": {
                            "privacyStatus": "public",
                            "selfDeclaredMadeForKids": False,
                        },
                        "contentDetails": {
                            "enableAutoStart": True,
                            "enableAutoStop": True,
                            "latencyPreference": "ultraLow",
                            "enableDvr": False,
                            "projection": "rectangular",
                            "enableLiveChat": True,
                        },
                    },
                )
                .execute()
            )

            self.broadcast_id = broadcast_response["id"]
            logger.info(f"✅ Created broadcast: {self.broadcast_id}")

            return self.broadcast_id

        except HttpError as e:
            logger.error(f"Broadcast creation error: {e}")
            raise

    def create_stream(self, video_height: int, video_framerate: int) -> str:
        """Create stream"""
        try:
            stream_response = (
                self.youtube.liveStreams()
                .insert(
                    part="snippet,cdn,contentDetails,status",
                    body={
                        "snippet": {
                            "title": f"Stream {datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        },
                        "cdn": {
                            "frameRate": f"{video_framerate}fps",
                            "ingestionType": "rtmp",
                            "resolution": f"{video_height}p",
                        },
                        "contentDetails": {"isReusable": False},
                    },
                )
                .execute()
            )

            self.stream_id = stream_response["id"]
            self.stream_key = stream_response["cdn"]["ingestionInfo"]["streamName"]

            logger.info(f"✅ Created stream: {self.stream_id}")
            logger.info(f"📝 Stream key: {self.stream_key[:10]}...")

            return self.stream_key

        except HttpError as e:
            logger.error(f"Stream creation error: {e}")
            raise

    def bind_broadcast_to_stream(self) -> None:
        """Bind broadcast to stream"""
        try:
            self.youtube.liveBroadcasts().bind(
                part="id,contentDetails", id=self.broadcast_id, streamId=self.stream_id
            ).execute()

            logger.info("✅ Broadcast bound to stream")

        except HttpError as e:
            logger.error(f"Binding error: {e}")
            raise

    def wait_for_stream_active(self, timeout: int = 60) -> bool:
        """Wait for stream to become active"""
        logger.info("⏳ Waiting for stream to become active...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                stream = (
                    self.youtube.liveStreams()
                    .list(part="status", id=self.stream_id)
                    .execute()
                )

                if stream["items"]:
                    status = stream["items"][0]["status"]["streamStatus"]
                    logger.info(f"Stream status: {status}")
                    if status == "active":
                        logger.info("✅ Stream is active")
                        return True

                time.sleep(2)

            except Exception as e:
                logger.error(f"Stream status check error: {e}")

        return False

    def transition_to_live(self) -> None:
        """Transition broadcast to live"""
        try:
            broadcast = (
                self.youtube.liveBroadcasts()
                .list(part="status,snippet", id=self.broadcast_id)
                .execute()
            )

            if broadcast["items"]:
                current_status = broadcast["items"][0]["status"]["lifeCycleStatus"]

                if current_status == "live":
                    logger.info("✅ Broadcast already live")
                    self.live_chat_id = broadcast["items"][0]["snippet"].get(
                        "liveChatId"
                    )
                    logger.info(f"📝 Live chat ID: {self.live_chat_id}")
                    return

                self.youtube.liveBroadcasts().transition(
                    broadcastStatus="live", id=self.broadcast_id, part="status"
                ).execute()

                logger.info("✅ Broadcast transitioned to live")

                # Wait a moment for the transition to complete
                time.sleep(2)

                # Get live chat ID
                broadcast = (
                    self.youtube.liveBroadcasts()
                    .list(part="snippet", id=self.broadcast_id)
                    .execute()
                )

                if broadcast["items"]:
                    self.live_chat_id = broadcast["items"][0]["snippet"].get(
                        "liveChatId"
                    )
                    logger.info(f"📝 Live chat ID: {self.live_chat_id}")

        except HttpError as e:
            if "redundantTransition" in str(e):
                logger.info("✅ Broadcast already live")
                try:
                    broadcast = (
                        self.youtube.liveBroadcasts()
                        .list(part="snippet", id=self.broadcast_id)
                        .execute()
                    )

                    if broadcast["items"]:
                        self.live_chat_id = broadcast["items"][0]["snippet"].get(
                            "liveChatId"
                        )
                        logger.info(f"📝 Live chat ID: {self.live_chat_id}")
                except Exception as fetch_error:
                    logger.error(f"Failed to fetch live chat ID: {fetch_error}")
            else:
                logger.error(f"Transition error: {e}")
                raise

    def get_chat_messages_fast(self, page_token: str = None) -> Dict[str, Any]:
        """Ultra-fast optimized chat message fetching"""
        if not self.live_chat_id:
            return {}

        try:
            params = {
                "liveChatId": self.live_chat_id,
                "part": "snippet,authorDetails",
                "maxResults": 2000,
                "fields": "items(snippet(type,displayMessage,publishedAt),authorDetails/displayName),nextPageToken,pollingIntervalMillis",
            }

            if page_token:
                params["pageToken"] = page_token

            if not self._chat_request:
                self._chat_request = self.youtube.liveChatMessages()

            response = self._chat_request.list(**params).execute()

            messages = [
                {
                    "author": item["authorDetails"]["displayName"],
                    "message": item["snippet"]["displayMessage"],
                    "timestamp": item["snippet"]["publishedAt"],
                }
                for item in response.get("items", [])
                if item["snippet"]["type"] == "textMessageEvent"
            ]

            return {
                "messages": messages,
                "nextPageToken": response.get("nextPageToken"),
                "pollingIntervalMillis": response.get("pollingIntervalMillis", 500),
            }

        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return {}

    def stop_broadcast(self) -> None:
        """Stop the broadcast"""
        try:
            if self.broadcast_id:
                self.youtube.liveBroadcasts().transition(
                    broadcastStatus="complete", id=self.broadcast_id, part="status"
                ).execute()
                logger.info("✅ Broadcast stopped")

        except Exception as e:
            logger.error(f"Broadcast stop error: {e}") 