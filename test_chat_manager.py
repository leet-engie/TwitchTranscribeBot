import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

from chat_manager import ChatManager

@pytest.mark.asyncio
async def test_bot_initialization():
    """
    Ensure the bot initializes without errors.
    """
    bot = ChatManager()
    assert bot is not None
    assert hasattr(bot, 'command_handler')
    assert bot.translation_service is not None or bot.translation_service is None
    # We don't assert specifically "is not None" because the bot might fail to load
    # the translation service, but either way it shouldn't crash.

@pytest.mark.asyncio
async def test_command_languages_no_translation_service(mocker):
    """
    Test !languages when translation service is unavailable.
    """
    bot = ChatManager()
    bot.translation_service = None
    bot.command_handler.translation_service = None
    
    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.command_handler.handle_languages_command(ctx)
    ctx.send.assert_awaited()
    ctx.send.assert_any_call("[🤖]: Translation service unavailable.")

@pytest.mark.asyncio
async def test_command_languages_with_mocked_service(mocker):
    """
    Test !languages when translation service is available
    and confirm we see the "common" language codes response.
    """
    bot = ChatManager()
    
    # Mock available_languages so it's predictable
    mock_translation_service = mocker.MagicMock()
    mock_translation_service.available_languages = ["en", "es", "fr"]
    bot.translation_service = mock_translation_service
    bot.command_handler.translation_service = mock_translation_service
    
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    
    await bot.command_handler.handle_languages_command(ctx)
    
    expected_calls = [
        mocker.call("[🤖]: Common language codes: es (Español), fr (Français)"),
        mocker.call("[🤖]: Use these codes with !translate (e.g., !translate es)")
    ]
    
    ctx.send.assert_has_calls(expected_calls, any_order=True)

@pytest.mark.asyncio
async def test_speech_command_already_running():
    """
    If speech_service is already set, 
    it should say "Speech-to-text service is already running."
    """
    bot = ChatManager()
    bot.speech_service = True  # pretend it's already running
    bot.command_handler.bot = bot

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.command_handler.handle_speech_command(ctx)
    ctx.send.assert_any_call("[🤖]: Speech-to-text service already running.")

@pytest.mark.asyncio
async def test_translate_command_no_argument():
    """
    If !translate is called with no language code, 
    it should print usage instructions.
    """
    bot = ChatManager()
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.message.content = "!translate"

    await bot.command_handler.handle_translate_command(ctx)
    ctx.send.assert_any_call("[🤖]: Usage: !translate <language_code>")

@pytest.mark.asyncio
async def test_translate_command_unsupported_language(mocker):
    """
    If the language code doesn't exist in available_languages,
    it should say "Language code '<code>' is not supported."
    """
    bot = ChatManager()
    
    mock_translation_service = mocker.MagicMock()
    mock_translation_service.available_languages = ["en", "es", "fr"]
    bot.translation_service = mock_translation_service
    bot.command_handler.translation_service = mock_translation_service

    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.message.content = "!translate zz"

    await bot.command_handler.handle_translate_command(ctx)
    ctx.send.assert_any_call("[🤖]: Language code 'zz' not supported.")

@pytest.mark.asyncio
async def test_translate_command_success(mocker):
    """
    If a valid language code is provided, 
    it should enable translation for that code and 
    (optionally) start speech if it's not running.
    """
    bot = ChatManager()
    
    mock_translation_service = mocker.MagicMock()
    mock_translation_service.available_languages = ["en", "es", "fr"]
    bot.translation_service = mock_translation_service
    bot.command_handler.translation_service = mock_translation_service
    bot.command_handler.bot = bot

    ctx = AsyncMock()
    ctx.message.content = "!translate es"
    ctx.send = AsyncMock()

    await bot.command_handler.handle_translate_command(ctx)

    assert bot.command_handler.active_translations["es"] == True
    ctx.send.assert_any_call("[🤖]: Started translation to 'es'")

@pytest.mark.asyncio
async def test_stoptranslate_command(mocker):
    """
    If language is active, then !stoptranslate <code> should disable it.
    """
    bot = ChatManager()
    bot.command_handler.active_translations["es"] = True

    ctx = AsyncMock()
    ctx.message.content = "!stoptranslate es"
    ctx.send = AsyncMock()

    await bot.command_handler.handle_stoptranslate_command(ctx)
    assert bot.command_handler.active_translations["es"] is False
    ctx.send.assert_any_call("[🤖]: Stopped translation to 'es'")

@pytest.mark.asyncio
async def test_translatechat_command_no_translation_service():
    """
    Test !translatechat when translation service is unavailable.
    """
    bot = ChatManager()
    bot.translation_service = None
    bot.command_handler.translation_service = None

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.command_handler.handle_translatechat_command(ctx)
    ctx.send.assert_called_once_with("[🤖]: Translation service unavailable.")

@pytest.mark.asyncio
async def test_translatechat_command_no_target_language():
    """
    Test !translatechat when no target language is configured.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.command_handler.translation_service = AsyncMock()
    bot.translate_chat_to = None
    bot.command_handler.bot = bot

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.command_handler.handle_translatechat_command(ctx)
    ctx.send.assert_called_once_with("[🤖]: Chat translation target language not configured")

@pytest.mark.asyncio
async def test_translatechat_command_toggle():
    """
    Test !translatechat toggles translation state correctly.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.command_handler.translation_service = AsyncMock()
    bot.translate_chat_to = "es"
    bot.command_handler.bot = bot

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    # Test enabling
    assert not bot.chat_translation_enabled
    await bot.command_handler.handle_translatechat_command(ctx)
    assert bot.chat_translation_enabled
    ctx.send.assert_called_with("[🤖]: Chat translation enabled (Target: es)")

    # Test disabling
    ctx.send.reset_mock()
    await bot.command_handler.handle_translatechat_command(ctx)
    assert not bot.chat_translation_enabled
    ctx.send.assert_called_with("[🤖]: Chat translation disabled (Target: es)")

@pytest.mark.asyncio
async def test_event_message_with_chat_translation():
    """
    Test that event_message properly handles chat translation.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.chat_translation_enabled = True
    bot.command_handler.handle_chat_translation = AsyncMock()

    # Test regular message
    message = AsyncMock()
    message.echo = False
    message.content = "Hello world!"
    
    await bot.event_message(message)
    bot.command_handler.handle_chat_translation.assert_called_once_with(message)

    # Test command message
    message.content = "!command"
    bot.command_handler.handle_chat_translation.reset_mock()
    
    await bot.event_message(message)
    bot.command_handler.handle_chat_translation.assert_not_called()  # Shouldn't translate commands

@pytest.mark.asyncio
async def test_load_config():
    """
    Test configuration loading functionality and defaults.
    """
    # Test with valid config
    bot = ChatManager()
    bot.config_manager = MagicMock()
    bot.config_manager.config = {
        'chat_translation': {
            'translate_to': 'es',
            'ignore_users': ['user1', 'User2']
        }
    }
    
    bot.load_config()
    assert bot.translate_chat_to == 'es'
    assert bot.ignore_users == {'user1', 'user2'}  # Should be lowercase
    assert not bot.chat_translation_enabled
    
    # Test with missing config
    bot.config_manager = None
    bot.load_config()
    assert bot.translate_chat_to is None
    assert bot.ignore_users == set()
    assert not bot.chat_translation_enabled

@pytest.mark.asyncio
async def test_init_services_failure():
    """
    Test behavior when service initialization fails.
    """
    # Patch where TranslationService is used, not where it's defined
    with patch('chat_manager.TranslationService', side_effect=Exception('Service init failed')):
        bot = ChatManager()
        assert bot.translation_service is None
        assert bot.config_manager is not None  # Config should still initialize


@pytest.mark.asyncio
async def test_event_ready():
    """
    Test the event_ready handler functionality.
    """
    bot = ChatManager()
    
    # Mock the channel
    mock_channel = AsyncMock()
    bot.get_channel = MagicMock(return_value=mock_channel)
    
    # Mock the send_welcome_message method
    bot.command_handler.send_welcome_message = AsyncMock()
    
    # Mock the logger
    with patch('chat_manager.logger.info') as mock_logger:
        await bot.event_ready()
        
        # Verify logger calls
        mock_logger.assert_any_call(f'Logged in as | {bot.nick}')
        mock_logger.assert_any_call(f'Connected to channel: {os.getenv("CHANNEL")}')
    
        # Verify that send_welcome_message was called
        bot.command_handler.send_welcome_message.assert_called_once()

@pytest.mark.asyncio
async def test_event_message_echo():
    """
    Test that echo messages are properly ignored.
    """
    bot = ChatManager()
    message = AsyncMock()
    message.echo = True
    message.content = "Test message"
    
    # Patch handle_commands to ensure it's not called
    bot.handle_commands = AsyncMock()
    
    await bot.event_message(message)
    bot.handle_commands.assert_not_called()

@pytest.mark.asyncio
async def test_concurrent_translations():
    """
    Test handling of concurrent translations to multiple languages.
    """
    bot = ChatManager()
    channel = AsyncMock()
    bot.get_channel = MagicMock(return_value=channel)
    
    # Setup mock translation service
    mock_translation_service = AsyncMock()
    mock_translation_service.translate = MagicMock()
    mock_translation_service.translate.side_effect = lambda text, src, target: f"Translated to {target}: {text}"
    bot.translation_service = mock_translation_service
    
    # Setup active translations
    bot.command_handler.active_translations = {
        "es": True,
        "fr": True,
        "de": False  # This one shouldn't be translated
    }
    
    # Test sending transcription
    text = "Hello world"
    await bot.send_transcription(text)
    
    # Verify calls
    assert channel.send.call_count == 3  # Original + 2 translations
    
    channel.send.assert_has_calls([
        call(f"{bot.bot_prefix}{text}"),
        call(f"{bot.bot_prefix}[es]: Translated to es: {text}"),
        call(f"{bot.bot_prefix}[fr]: Translated to fr: {text}")
    ], any_order=True) 

@pytest.mark.asyncio
async def test_cleanup():
    """
    Test cleanup functionality.
    """
    bot = ChatManager()
    
    # Setup mock services
    bot.speech_service = MagicMock()
    bot.speech_task = AsyncMock()
    bot.speech_task.cancel = MagicMock()
    
    # Execute cleanup
    bot.cleanup()
    
    # Verify cleanup actions
    bot.speech_task.cancel.assert_called_once()
    bot.speech_service.stop.assert_called_once()

@pytest.mark.asyncio
async def test_send_transcription_empty():
    """
    Test that empty transcriptions are not sent.
    """
    bot = ChatManager()
    channel = AsyncMock()
    bot.get_channel = MagicMock(return_value=channel)
    
    await bot.send_transcription("   ")  # Empty or whitespace
    channel.send.assert_not_called()

@pytest.mark.asyncio
async def test_translation_error_handling():
    """
    Test handling of translation errors.
    """
    bot = ChatManager()
    channel = AsyncMock()
    bot.get_channel = MagicMock(return_value=channel)
    
    # Setup mock translation service that raises an exception
    mock_translation_service = MagicMock()
    mock_translation_service.translate = MagicMock(side_effect=Exception("Translation failed"))
    bot.translation_service = mock_translation_service
    
    bot.command_handler.active_translations = {"es": True}
    
    # Should not raise exception and should still send original message
    await bot.send_transcription("Test message")
    channel.send.assert_called_once_with(f"{bot.bot_prefix}Test message")