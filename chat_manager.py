from twitchio.ext import commands
import os
import logging
import asyncio
import signal
import sys
from typing import Optional, Dict
from dotenv import load_dotenv
from translation_service import TranslationService

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
        
        # Speech and translation services tracking
        self.speech_service: Optional[object] = None    # Will store the AudioTranscriber instance
        self.speech_task: Optional[asyncio.Task] = None # The async Task for speech-to-text
        self.translation_service: Optional[TranslationService] = None
        
        # Track which languages are currently active for translation
        # e.g.: {'es': True, 'fr': True}
        self.active_translations: Dict[str, bool] = {}
        
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

        # Initialize translation service
        try:
            self.translation_service = TranslationService()
            logger.info("Translation service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            self.translation_service = None

    def get_language_examples(self) -> str:
        """Get a string of some common language codes and names."""
        if not self.translation_service:
            return ""
        common_languages = {
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'pt': 'Portugese',
            'it': 'Italian',
            'pl': 'Polish',
            'nl': 'Dutch',
            'ko': 'Korean',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese'
        }
        return ", ".join([
            f"{code} ({name})" for code, name in common_languages.items()
            if code in self.translation_service.available_languages
        ])

    def chunk_message(self, content: str, max_len: int = 500) -> list[str]:
        """
        Helper method to split a long message into <= max_len pieces.
        Twitch enforces a 500-character limit per message.
        """
        # We don't want to break mid-word if possible, but simplest approach is just a raw slice
        # For a more advanced approach, you could split by spaces near max_len
        chunks = []
        start = 0
        while start < len(content):
            chunk = content[start:start+max_len]
            chunks.append(chunk)
            start += max_len
        return chunks

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
        await channel.send(f"{self.bot_prefix}!translate <code> - Start translation to specified language (can enable multiple)")
        await channel.send(f"{self.bot_prefix}!stoptranslate <code> - Stop translation for that language")
        await channel.send(f"{self.bot_prefix}!languages - List supported language codes")

    async def event_message(self, message):
        """Handle incoming messages."""
        if message.echo:
            return

        # Log incoming messages
        logger.debug(f'{message.author.name}: {message.content}')
        
        # Handle commands
        await self.handle_commands(message)

    async def send_transcription(self, text: str):
        """
        Send transcribed text to chat with translations if active.
        Called by the speech_service whenever new text is recognized.
        """
        if not text.strip():
            return
            
        channel = self.get_channel(os.getenv('CHANNEL'))
        
        # 1) Send the original transcription
        await channel.send(f"{self.bot_prefix}{text}")
        
        # 2) Send translations if any are active
        if self.translation_service and self.active_translations:
            for target_lang, is_active in self.active_translations.items():
                if is_active:
                    try:
                        # If your source language is not English, adjust "en" as needed
                        translated_text = self.translation_service.translate(text, "en", target_lang)
                        await channel.send(f"{self.bot_prefix}[{target_lang}]: {translated_text}")
                    except Exception as e:
                        logger.error(f"Translation error for {target_lang}: {e}")

    @commands.command(name='languages')
    async def languages_command(self, ctx: commands.Context):
        """List supported language codes."""
        if not self.translation_service:
            await ctx.send(f"{self.bot_prefix}Translation service is not available.")
            return

        # 1) Send common examples
        common_examples = self.get_language_examples()
        await ctx.send(f"{self.bot_prefix}Common language codes: {common_examples}")

        # 3) Split into multiple <=500-char chunks, then send each
        chunks = self.chunk_message(full_list_str, max_len=500)
        for chunk in chunks:
            await ctx.send(chunk)

        # 4) Final usage note
        await ctx.send(f"{self.bot_prefix}Use these codes with the !translate command (e.g., !translate es)")

    @commands.command(name='speech')
    async def speech_command(self, ctx: commands.Context):
        """Handle speech service command."""
        try:
            if not self.speech_service:
                try:
                    # Make sure you have updated 'speech_to_chat.py' so that
                    # AudioTranscriber.record_and_transcribe() is truly asynchronous
                    from speech_to_chat import AudioTranscriber
                    
                    self.speech_service = AudioTranscriber(self)
                    
                    # Start the transcription in an async task
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

    @commands.command(name='translate')
    async def translate_command(self, ctx: commands.Context):
        """Enable translation to the specified language code."""
        try:
            parts = ctx.message.content.split()
            if len(parts) < 2:
                await ctx.send(f"{self.bot_prefix}Usage: !translate <language_code>")
                await ctx.send(f"{self.bot_prefix}Use !languages to see available language codes")
                return

            target_lang = parts[1].lower()

            if not self.translation_service:
                await ctx.send(f"{self.bot_prefix}Translation service is not available.")
                return

            # Validate language
            if target_lang not in self.translation_service.available_languages:
                common_examples = self.get_language_examples()
                await ctx.send(f"{self.bot_prefix}Language code '{target_lang}' is not supported.")
                await ctx.send(f"{self.bot_prefix}Common language codes: {common_examples}")
                await ctx.send(f"{self.bot_prefix}Use !languages to see all available codes")
                return

            # Start speech service if not already running
            if not self.speech_service:
                await self.speech_command(ctx)
            
            # Mark this translation as active
            if self.active_translations.get(target_lang, False):
                await ctx.send(f"{self.bot_prefix}Translation to '{target_lang}' is already active.")
            else:
                self.active_translations[target_lang] = True
                await ctx.send(f"{self.bot_prefix}Started translation to '{target_lang}'")

        except Exception as e:
            logger.error(f"Error in translate command: {e}")
            await ctx.send(f"{self.bot_prefix}Error processing translate command")

    @commands.command(name='stoptranslate')
    async def stoptranslate_command(self, ctx: commands.Context):
        """Disable translation for the specified language code."""
        try:
            parts = ctx.message.content.split()
            if len(parts) < 2:
                await ctx.send(f"{self.bot_prefix}Usage: !stoptranslate <language_code>")
                return

            target_lang = parts[1].lower()

            if target_lang in self.active_translations and self.active_translations[target_lang]:
                self.active_translations[target_lang] = False
                await ctx.send(f"{self.bot_prefix}Stopped translation to '{target_lang}'")
            else:
                await ctx.send(f"{self.bot_prefix}Translation to '{target_lang}' is not active.")

        except Exception as e:
            logger.error(f"Error in stoptranslate command: {e}")
            await ctx.send(f"{self.bot_prefix}Error processing stoptranslate command")

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
