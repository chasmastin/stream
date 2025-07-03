#!/usr/bin/env python3
"""
Script to generate a new YouTube refresh token with correct scopes
Run this script and follow the instructions to get a new refresh token.
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Your current credentials
CLIENT_ID = ""
CLIENT_SECRET = ""

# Scopes for YouTube live streaming
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl"
]

def get_new_refresh_token():
    """Get a new refresh token"""
    
    # Create credentials dict for the flow
    client_config = {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8080"]
        }
    }
    
    try:
        # Create the flow
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        
        # Run local server to handle OAuth callback
        credentials = flow.run_local_server(port=8080)
        
        print("\n" + "="*60)
        print("SUCCESS! Your new refresh token is:")
        print("="*60)
        print(f"REFRESH_TOKEN = '{credentials.refresh_token}'")
        print("="*60)
        print("\nUpdate your modules/config.py file with this new refresh token.")
        print("Replace the current YOUTUBE_REFRESH_TOKEN value with the one above.")
        
        # Test the credentials
        youtube = build('youtube', 'v3', credentials=credentials)
        response = youtube.channels().list(part='snippet', mine=True).execute()
        
        if response['items']:
            channel_name = response['items'][0]['snippet']['title']
            print(f"\n✅ Successfully authenticated for channel: {channel_name}")
        
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have google-auth-oauthlib installed:")
        print("   pip install google-auth-oauthlib")
        print("2. Make sure your OAuth credentials are correct")
        print("3. Allow the app to access your YouTube account when prompted")

if __name__ == "__main__":
    print("🔑 YouTube Refresh Token Generator")
    print("="*40)
    print("This will open a browser window for authentication.")
    print("Make sure to allow access to your YouTube account.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    get_new_refresh_token() 