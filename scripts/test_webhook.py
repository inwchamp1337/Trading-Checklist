from dotenv import load_dotenv
import os

# Load .env file at the beginning
load_dotenv()

from discord_notify import send_discord_webhook

# Send test message
send_discord_webhook('Test webhook message from test_webhook.py')
print("Message sent successfully!")