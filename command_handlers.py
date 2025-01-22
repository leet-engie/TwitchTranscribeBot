import os
import logging
from typing import Dict, Optional
import asyncio
from speech_service import SpeechService

logger = logging.getLogger(__name__)

class CommandHandler:
    def __init__(self, bot):
        self.bot = bot
        self.active_translations: Dict[str, bool] = {}

    async def send_welcome_message(self):
        """Send welcome message with available commands."""
        channel = self.bot.get_channel(os.getenv('CHANNEL'))
        await channel.send(f"{self.bot.bot_prefix}Chat manager connected!")
        await channel.send(f"{self.bot.bot_prefix}Available commands:")
        await channel.send(f"{self.bot.bot_prefix}!speech - Start speech-to-text service")
        await channel.send(f"{self.bot.bot_prefix}!translate <code> - Start translation")
        await channel.send(f"{self.bot.bot_prefix}!stoptranslate <code> - Stop translation")
        await channel.send(f"{self.bot.bot_prefix}!languages - List supported languages")
        await channel.send(f"{self.bot.bot_prefix}!translatechat - Toggle chat translation")

    def get_language_examples(self) -> str:
        """Get a string of common language codes and names."""
        if not self.bot.translation_service:
            return ""
        common_languages = {
            'es': 'Español', 'fr': 'Français', 'de': 'Deutsch', 
            'pt': 'Português', 'it': 'Italiano', 'pl': 'Polski',
            'nl': 'Nederlands', 'ko': '한국어', 'ru': 'Русский',
            'zh': '中文', 'ja': '日本語'
        }
        return ", ".join([
            f"{code} ({name})" for code, name in common_languages.items()
            if code in self.bot.translation_service.available_languages
        ])

    async def handle_chat_translation(self, message):
        """Handle translation of chat messages."""
        try:
            if message.author.name.lower() in self.bot.ignore_users:
                return

            try:
                translated_text = await self._translate_message(message.content)
                if translated_text:
                    channel = self.bot.get_channel(os.getenv('CHANNEL'))
                    await channel.send(
                        f"{self.bot.bot_prefix}[{message.author.name}]"
                        f"[{self.bot.translate_chat_to}]: {translated_text}")

            except Exception as e:
                logger.error(f"Translation error: {e}")

        except Exception as e:
            logger.error(f"Error in chat translation: {e}")

    async def _translate_message(self, content: str) -> Optional[str]:
        """Translate a message with fallback handling."""
        try:
            if self.bot.translate_chat_to == 'en':
                translated_text = self.bot.translation_service.translate(
                    content, 'es', 'en')
            else:
                translated_text = self.bot.translation_service.model.translate(
                    content, target_lang=self.bot.translate_chat_to)

            # Compare cleaned versions
            cleaned_original = ' '.join(content.lower().split())
            cleaned_translation = ' '.join(translated_text.lower().split())
            
            return translated_text if cleaned_translation != cleaned_original else None

        except Exception:
            # Fallback translation attempt
            try:
                source_lang = 'es' if self.bot.translate_chat_to == 'en' else 'en'
                translated_text = self.bot.translation_service.translate(
                    content, source_lang, self.bot.translate_chat_to)
                
                return translated_text if translated_text.lower() != content.lower() else None
                
            except Exception as fallback_error:
                logger.error(f"Translation fallback error: {fallback_error}")
                return None

    async def handle_languages_command(self, ctx):
        """Handle the !languages command."""
        if not self.bot.translation_service:
            await ctx.send(f"{self.bot.bot_prefix}Translation service unavailable.")
            return

        common_examples = self.get_language_examples()
        await ctx.send(f"{self.bot.bot_prefix}Common language codes: {common_examples}")
        await ctx.send(
            f"{self.bot.bot_prefix}Use these codes with !translate "
            f"(e.g., !translate es)")

    async def handle_translatechat_command(self, ctx):
        """Handle the !translatechat command."""
        if not self.bot.translation_service:
            await ctx.send(f"{self.bot.bot_prefix}Translation service unavailable.")
            return

        if not self.bot.translate_chat_to:
            await ctx.send(
                f"{self.bot.bot_prefix}Chat translation target language not configured")
            return

        self.bot.chat_translation_enabled = not self.bot.chat_translation_enabled
        status = "enabled" if self.bot.chat_translation_enabled else "disabled"
        await ctx.send(
            f"{self.bot.bot_prefix}Chat translation {status} "
            f"(Target: {self.bot.translate_chat_to})")

    async def handle_speech_command(self, ctx):
        """Handle the !speech command."""
        try:
            if not self.bot.speech_service:
                self.bot.speech_service = SpeechService(self.bot)
                self.bot.speech_task = asyncio.create_task(
                    self.bot.speech_service.record_and_transcribe()
                )
                await ctx.send(f"{self.bot.bot_prefix}Speech-to-text service started!")
            else:
                await ctx.send(
                    f"{self.bot.bot_prefix}Speech-to-text service already running.")
                    
        except Exception as e:
            logger.error(f"Error in speech command: {e}")
            await ctx.send(f"{self.bot.bot_prefix}Error processing speech command")

    async def handle_translate_command(self, ctx):
        """Handle the !translate command."""
        try:
            parts = ctx.message.content.split()
            if len(parts) < 2:
                await ctx.send(f"{self.bot.bot_prefix}Usage: !translate <language_code>")
                await ctx.send(f"{self.bot.bot_prefix}Use !languages to see available codes")
                return

            target_lang = parts[1].lower()

            if not self.bot.translation_service:
                await ctx.send(f"{self.bot.bot_prefix}Translation service unavailable.")
                return

            if target_lang not in self.bot.translation_service.available_languages:
                common_examples = self.get_language_examples()
                await ctx.send(
                    f"{self.bot.bot_prefix}Language code '{target_lang}' not supported.")
                await ctx.send(f"{self.bot.bot_prefix}Common codes: {common_examples}")
                return

            if not self.bot.speech_service:
                await self.handle_speech_command(ctx)
            
            if self.active_translations.get(target_lang, False):
                await ctx.send(
                    f"{self.bot.bot_prefix}Translation to '{target_lang}' already active.")
            else:
                self.active_translations[target_lang] = True
                await ctx.send(f"{self.bot.bot_prefix}Started translation to '{target_lang}'")

        except Exception as e:
            logger.error(f"Error in translate command: {e}")
            await ctx.send(f"{self.bot.bot_prefix}Error processing translate command")

    async def handle_stoptranslate_command(self, ctx):
        """Handle the !stoptranslate command."""
        try:
            parts = ctx.message.content.split()
            if len(parts) < 2:
                await ctx.send(f"{self.bot.bot_prefix}Usage: !stoptranslate <language_code>")
                return

            target_lang = parts[1].lower()

            if target_lang in self.active_translations and self.active_translations[target_lang]:
                self.active_translations[target_lang] = False
                await ctx.send(f"{self.bot.bot_prefix}Stopped translation to '{target_lang}'")
            else:
                await ctx.send(
                    f"{self.bot.bot_prefix}Translation to '{target_lang}' not active.")

        except Exception as e:
            logger.error(f"Error in stoptranslate command: {e}")
            await ctx.send(f"{self.bot.bot_prefix}Error processing stoptranslate command")
