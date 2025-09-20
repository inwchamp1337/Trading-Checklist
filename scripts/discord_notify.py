import os
import asyncio
import logging

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    discord = None

logger = logging.getLogger(__name__)


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
                await channel.send(content)
            except Exception as exc:
                logger.exception("Failed to send discord message: %s", exc)
            finally:
                # Close client properly
                if not client.is_closed():
                    await client.close()

        await client.start(token)
    except Exception as exc:
        logger.exception("Failed to start discord client: %s", exc)
        raise
    finally:
        # Ensure client is closed and wait for cleanup
        if not client.is_closed():
            await client.close()
        # Give time for connections to close
        await asyncio.sleep(0.1)


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

    # Create new event loop for each call to avoid conflicts
    try:
        # Try to use existing loop if available
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new one
            import threading
            result = [None]
            exception = [None]
            
            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result[0] = new_loop.run_until_complete(_send(token, channel_id, content))
                except Exception as e:
                    exception[0] = e
                finally:
                    new_loop.close()
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception[0]:
                raise exception[0]
        else:
            loop.run_until_complete(_send(token, channel_id, content))
    except RuntimeError:
        # No event loop exists, create a new one
        asyncio.run(_send(token, channel_id, content))


