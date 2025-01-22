from twitchio.ext import commands
import os
import logging
import asyncio
import signal
import sys
from typing import Optional, Dict
from dotenv import load_dotenv
from translation_service import TranslationService
from config_manager import ConfigManager
from command_handler import CommandHandler
from speech_service import SpeechService

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
        self.speech_service: Optional[SpeechService] = None
        self.speech_task: Optional[asyncio.Task] = None
        self.translation_service: Optional[TranslationService] = None
        
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

        # Initialize services
        self.init_services()
        
        # Initialize command handler
        self.command_handler = CommandHandler(self)
        
        # Register commands
        self.register_commands()

    def init_services(self):
        """Initialize all required services."""
        # Initialize config
        try:
            self.config_manager = ConfigManager()
            self.load_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config_manager = None

        # Initialize translation service
        try:
            self.translation_service = TranslationService()
            logger.info("Translation service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            self.translation_service = None

    def load_config(self):
        """Load configuration from config manager."""
        if self.config_manager:
            self.chat_translation_config = self.config_manager.config.get('chat_translation', {})
            self.translate_chat_to = self.chat_translation_config.get('translate_to')
            self.ignore_users = set(username.lower() for username in 
                                  self.chat_translation_config.get('ignore_users', []))
            self.chat_translation_enabled = False
        else:
            self.translate_chat_to = None
            self.ignore_users = set()
            self.chat_translation_enabled = False

    def register_commands(self):
        """Register all bot commands."""
        @self.command(name='languages')
        async def languages_command(ctx):
            await self.command_handler.handle_languages_command(ctx)

        @self.command(name='translatechat')
        async def translatechat_command(ctx):
            await self.command_handler.handle_translatechat_command(ctx)

        @self.command(name='speech')
        async def speech_command(ctx):
            await self.command_handler.handle_speech_command(ctx)

        @self.command(name='translate')
        async def translate_command(ctx):
            await self.command_handler.handle_translate_command(ctx)

        @self.command(name='stoptranslate')
        async def stoptranslate_command(ctx):
            await self.command_handler.handle_stoptranslate_command(ctx)

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
        await self.command_handler.send_welcome_message()

    async def event_message(self, message):
        """Handle incoming messages."""
        if message.echo:
            return

        # Log incoming messages
        logger.debug(f'{message.author.name}: {message.content}')

        # Handle chat translation if enabled
        if (self.chat_translation_enabled and self.translation_service and 
            message.content and not message.content.startswith('!')):
            await self.command_handler.handle_chat_translation(message)
        
        # Handle commands
        await self.handle_commands(message)

    async def send_transcription(self, text: str):
        """Send transcribed text to chat with translations."""
        if not text.strip():
            return
            
        channel = self.get_channel(os.getenv('CHANNEL'))
        
        # Send original transcription
        await channel.send(f"{self.bot_prefix}{text}")
        
        # Send translations if any are active
        if self.translation_service and self.command_handler.active_translations:
            for target_lang, is_active in self.command_handler.active_translations.items():
                if is_active:
                    try:
                        translated_text = self.translation_service.translate(text, "en", target_lang)
                        if translated_text:
                            await channel.send(f"{self.bot_prefix}[{target_lang}]: {translated_text}")
                    except Exception as e:
                        logger.error(f"Translation error for {target_lang}: {e}")

    def cleanup(self):
        """Clean up resources on shutdown."""
        logger.info("Cleaning up resources...")
        if self.speech_task:
            self.speech_task.cancel()
        if self.speech_service:
            self.speech_service.stop()
        logger.info("Cleanup complete.")