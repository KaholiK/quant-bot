"""
Discord bot runner for quant trading control.
"""

import asyncio
import os

from loguru import logger


def run_discord_bot():
    """Run Discord bot if token is configured."""
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        print("""
🤖 Discord Bot Setup Instructions:

1. Create a Discord application: https://discord.com/developers/applications
2. Create a bot and copy the token
3. Set environment variable: DISCORD_BOT_TOKEN=your_token_here
4. Add ALLOWED_USER_IDS=user_id1,user_id2 (comma-separated Discord user IDs)
5. Optional: Add DISCORD_WEBHOOK_URL for passive alerts
6. Invite bot to your server with slash command permissions

Example .env file:
DISCORD_BOT_TOKEN=your_bot_token_here
ALLOWED_USER_IDS=123456789012345678
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ADMIN_TOKEN=your_admin_api_token

Then run: python -m bots
        """)
        return

    try:
        from .discord_bot import main
        asyncio.run(main())
    except ImportError as e:
        logger.error(f"Failed to import discord bot: {e}")
        logger.info("Install discord.py: pip install discord.py")
    except Exception as e:
        logger.error(f"Failed to run discord bot: {e}")


if __name__ == "__main__":
    run_discord_bot()
