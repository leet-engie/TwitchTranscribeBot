from twitchio.ext import commands
import os
import logging
import asyncio
import signal
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatManager(commands.Bot):
    def __init__(self):
        """Initialize the chat manager."""
        # Command prefix for bot messages
        self.bot_prefix = "[🤖]: "
        
        # Speech service tracking
        self.speech_service: Optional[object] = None  # Will hold speech service instance
        self.speech_task: Optional[asyncio.Task] = None
        
        # Initialize parent bot class
        super().__init__(
            token=os.getenv('TMI_TOKEN'),
            prefix='!',
            nick=os.getenv('BOT_NICK'),
            initial_channels=[os.getenv('CHANNEL')]
        )
        
        # Setup shutdown handlers
        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received...")
        self.should_stop = True
        self.cleanup()
        sys.exit(0)

    async def event_ready(self):
        """Handle bot ready event."""
        logger.info(f'Logged in as | {self.nick}')
        logger.info(f'Connected to channel: {os.getenv("CHANNEL")}')
        
        channel = self.get_channel(os.getenv('CHANNEL'))
        await channel.send(f"{self.bot_prefix}Chat manager connected!")
        await channel.send(f"{self.bot_prefix}Available commands:")
        await channel.send(f"{self.bot_prefix}!speech - Start speech-to-text service")
        await channel.send(f"{self.bot_prefix}!speech stop - Stop speech-to-text service")

    async def event_message(self, message):
        """Handle incoming messages."""
        if message.echo:
            return

        # Log incoming messages
        logger.debug(f'{message.author.name}: {message.content}')
        
        # Handle commands
        await self.handle_commands(message)

    async def send_transcription(self, text: str):
        """Send transcribed text to the chat."""
        if text.strip():  # Only send non-empty messages
            channel = self.get_channel(os.getenv('CHANNEL'))
            await channel.send(f"{self.bot_prefix}{text}")

    @commands.command(name='speech')
    async def speech_command(self, ctx: commands.Context):
        """Handle speech service commands."""
        try:
            parts = ctx.message.content.split()
            stop_service = len(parts) > 1 and parts[1].lower() == 'stop'
            
            if stop_service:
                if self.speech_service:
                    # Stop the speech service
                    if self.speech_task:
                        self.speech_task.cancel()
                    self.speech_service.stop()
                    self.speech_service = None
                    await ctx.send(f"{self.bot_prefix}Speech-to-text service stopped.")
                else:
                    await ctx.send(f"{self.bot_prefix}Speech-to-text service is not running.")
            else:
                if not self.speech_service:
                    try:
                        # Import here to avoid circular imports
                        from speech_to_chat import AudioTranscriber
                        
                        # Create speech service with bot instance
                        self.speech_service = AudioTranscriber(self)
                        self.speech_task = asyncio.create_task(
                            self.speech_service.record_and_transcribe()
                        )
                        
                        await ctx.send(f"{self.bot_prefix}Speech-to-text service started!")
                    except Exception as e:
                        logger.error(f"Error starting speech service: {e}")
                        await ctx.send(f"{self.bot_prefix}Failed to start speech-to-text service.")
                else:
                    await ctx.send(f"{self.bot_prefix}Speech-to-text service is already running.")
                    
        except Exception as e:
            logger.error(f"Error in speech command: {e}")
            await ctx.send(f"{self.bot_prefix}Error processing speech command")

    def cleanup(self):
        """Clean up resources on shutdown."""
        logger.info("Cleaning up resources...")
        if self.speech_task:
            self.speech_task.cancel()
        if self.speech_service:
            self.speech_service.stop()
        logger.info("Cleanup complete.")

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