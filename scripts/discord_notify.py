import os
import requests
import logging

LOG = logging.getLogger("discord_notify")
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

# Global buffer for accumulated messages
_message_buffer = []

def send_discord_message(content: str) -> None:
    """ส่งข้อความผ่าน webhook (ไม่ต้อง login)"""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DISCORD_WEBHOOK_URL must be set in env")
    
    payload = {"content": content}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        LOG.info("Message sent successfully")
    except Exception as e:
        LOG.exception(f"Failed to send Discord message: {e}")
        raise

def send_discord_webhook(content: str) -> None:
    """Alias for send_discord_message - sends message using webhook."""
    send_discord_message(content)

def accumulate_message(content: str) -> None:
    """Add message to buffer for bulk sending later."""
    global _message_buffer
    _message_buffer.append(content)

def send_bulk_discord_message() -> None:
    """Send accumulated messages as a single batch."""
    global _message_buffer
    if not _message_buffer:
        return
    
    # Join all messages with separator
    combined = "\n\n---\n\n".join(_message_buffer)
    
    # Clear buffer
    _message_buffer.clear()
    
    # Send the combined message
    send_discord_message(combined)


