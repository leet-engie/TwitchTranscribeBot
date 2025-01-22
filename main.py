# main.py
import asyncio
import logging
import sys
from chat_manager import ChatManager  # Add this import

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the chat manager."""
    bot = ChatManager()
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        if bot:
            bot.cleanup()
        logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Forced exit requested...")
        sys.exit(0)