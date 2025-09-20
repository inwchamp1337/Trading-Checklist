import os
import asyncio
import logging

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    discord = None

logger = logging.getLogger(__name__)

# Global message accumulator
_message_buffer = []
_max_message_length = 1900  # Discord limit is 2000, leave some margin


async def _send(token: str, channel_id: int, content: str) -> None:
    if discord is None:
        raise RuntimeError("discord.py is not installed")

    intents = discord.Intents.none()
    client = discord.Client(intents=intents)

    try:
        @client.event
        async def on_ready():
            try:
                channel = client.get_channel(channel_id)
                if channel is None:
                    channel = await client.fetch_channel(channel_id)
                
                # Split long messages if needed
                if len(content) <= 2000:
                    await channel.send(content)
                else:
                    # Split into chunks
                    chunks = [content[i:i+_max_message_length] for i in range(0, len(content), _max_message_length)]
                    for chunk in chunks:
                        await channel.send(chunk)
                        await asyncio.sleep(0.5)  # Rate limit between chunks
                        
            except Exception as exc:
                logger.exception("Failed to send discord message: %s", exc)
            finally:
                if not client.is_closed():
                    await client.close()

        await client.start(token)
    except Exception as exc:
        logger.exception("Failed to start discord client: %s", exc)
        raise
    finally:
        if not client.is_closed():
            await client.close()
        await asyncio.sleep(0.1)


def accumulate_message(content: str) -> None:
    """Add a message to the buffer for bulk sending later."""
    global _message_buffer
    _message_buffer.append(content)


def send_bulk_discord_message() -> None:
    """Send all accumulated messages as one bulk message."""
    global _message_buffer
    if not _message_buffer:
        return
    
    # Combine all messages with separators
    combined = "\n\n---\n\n".join(_message_buffer)
    
    # Clear buffer
    _message_buffer.clear()
    
    # Send combined message
    send_discord_message(combined)


def send_discord_message(content: str) -> None:
    """Send a short message to the channel configured via env vars."""
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL_ID")
    if not token or not channel:
        raise RuntimeError("DISCORD_TOKEN and DISCORD_CHANNEL_ID must be set in env")

    try:
        channel_id = int(channel)
    except ValueError:
        raise RuntimeError("DISCORD_CHANNEL_ID must be an integer channel id")

    # Use asyncio.run for proper cleanup
    asyncio.run(_send(token, channel_id, content))


