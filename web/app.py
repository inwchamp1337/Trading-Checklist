import os
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
from flask import Flask, jsonify, render_template
from pathlib import Path
import sys

# Add repo root to path to import scripts
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import discord
except ImportError:
    discord = None

LOG = logging.getLogger("discord_web_fetcher")

async def fetch_discord_messages(token: str, channel_id: int) -> list:
    """Fetches messages from a Discord channel for today."""
    if discord is None:
        raise RuntimeError("discord.py is not installed.")

    th_tz = ZoneInfo("Asia/Bangkok")
    now_th = datetime.now(th_tz)
    start_of_day = datetime(now_th.year, now_th.month, now_th.day, 0, 0, 0, tzinfo=th_tz)

    intents = discord.Intents.default()
    intents.messages = True
    client = discord.Client(intents=intents)
    messages_data = []

    try:
        await client.login(token)
        channel = await client.fetch_channel(channel_id)
        if not channel:
            LOG.error(f"Channel {channel_id} not found")
            return []

        # Fetch last 100 messages since the start of today
        async for msg in channel.history(limit=100, after=start_of_day):
            # created_at is timezone-naive in UTC -> treat as UTC then convert
            created_utc = msg.created_at.replace(tzinfo=ZoneInfo("UTC"))
            # Keep ISO format in UTC for the API; client will localize to Asia/Bangkok
            messages_data.append({
                "timestamp": created_utc.isoformat(),
                "author": msg.author.display_name,
                "content": msg.content
            })
    except Exception as e:
        LOG.error(f"Failed to fetch messages: {e}")
    finally:
        if not client.is_closed():
            await client.close()
    
    # Return newest first
    return sorted(messages_data, key=lambda x: x['timestamp'], reverse=True)

# --- Flask App ---
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/messages')
def get_messages():
    """API endpoint to get messages from Discord."""
    token = os.getenv("DISCORD_TOKEN")
    channel_id_str = os.getenv("DISCORD_CHANNEL_ID")

    if not token or not channel_id_str:
        return jsonify({"error": "DISCORD_TOKEN and DISCORD_CHANNEL_ID must be set"}), 500

    try:
        channel_id = int(channel_id_str)
        # Run the async function to get messages
        messages = asyncio.run(fetch_discord_messages(token, channel_id))
        return jsonify(messages)
    except Exception as e:
        LOG.exception("Error in /api/messages")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("WEB_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)