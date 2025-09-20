"""Small test utility to fetch messages from a channel for today (Asia/Bangkok)

Usage: set `DISCORD_TOKEN` and `DISCORD_CHANNEL_ID` in your environment and run:

    python scripts/discord_listener.py

This script uses discord.py's REST client to fetch message history and prints messages
that were created today in Asia/Bangkok timezone. It's intentionally minimal for quick testing.
"""
from __future__ import annotations
import os
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    discord = None

LOG = logging.getLogger("discord_listener")


async def fetch_today_messages(token: str, channel_id: int) -> None:
    if discord is None:
        raise RuntimeError("discord.py is not installed. Install with: pip install discord.py")

    # timezone: Bangkok
    th_tz = ZoneInfo("Asia/Bangkok")
    now_th = datetime.now(th_tz)
    start_of_day = datetime(now_th.year, now_th.month, now_th.day, 0, 0, 0, tzinfo=th_tz)

    intents = discord.Intents.none()
    # Use a ClientSession via discord.Client for REST fetches
    client = discord.Client(intents=intents)

    try:
        await client.login(token)
        channel = await client.fetch_channel(channel_id)
        if channel is None:
            print(f"Channel {channel_id} not found")
            return

        print(f"Fetching messages for channel {channel_id} since {start_of_day.isoformat()}")

        # discord.py message history is async iterator (needs permission to read message history)
        after = start_of_day
        messages = []
        async for msg in channel.history(limit=500, after=after):
            # convert msg.created_at (naive UTC) to timezone-aware
            created_utc = msg.created_at.replace(tzinfo=ZoneInfo("UTC"))
            created_th = created_utc.astimezone(th_tz)
            messages.append((created_th, msg.author.display_name, msg.content))

        if not messages:
            print("No messages found for today.")
            return

        # Print messages newest first
        for created_th, author, content in sorted(messages, key=lambda x: x[0]):
            ts = created_th.strftime('%Y-%m-%d %H:%M:%S %Z')
            print(f"[{ts}] {author}: {content}")

    finally:
        # ensure we close the client
        try:
            await client.close()
        except Exception:
            pass


def main() -> None:
    token = "MTQxNzA5MjQzNDcxMzc3MjA0Mg.GYUB3T.tggF9TfSFdsvkORRVyimSqrvB2EnOZu-QBiAZE"
    channel = "1417121359372484610"
    if not token or not channel:
        print("Please set DISCORD_TOKEN and DISCORD_CHANNEL_ID in the environment.")
        return

# DISCORD_TOKEN=MTQxNzA5MjQzNDcxMzc3MjA0Mg.GYUB3T.tggF9TfSFdsvkORRVyimSqrvB2EnOZu-QBiAZE
# DISCORD_CHANNEL_ID=1417121359372484610
    try:
        channel_id = int(channel)
    except ValueError:
        print("DISCORD_CHANNEL_ID must be an integer")
        return

    asyncio.run(fetch_today_messages(token, channel_id))


if __name__ == "__main__":
    main()
