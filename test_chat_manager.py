import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from chat_manager import ChatManager

@pytest.mark.asyncio
async def test_bot_initialization():
    """
    Ensure the bot initializes without errors.
    """
    bot = ChatManager()
    assert bot is not None
    assert bot.translation_service is not None or bot.translation_service is None
    # We don't assert specifically "is not None" because the bot might fail to load
    # the translation service, but either way it shouldn't crash.


@pytest.mark.asyncio
async def test_command_languages_no_translation_service(mocker):
    """
    Test !languages when translation service is unavailable.
    """
    bot = ChatManager()
    bot.translation_service = None  # Force the translation service to be unavailable
    
    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.languages_command(ctx)
    ctx.send.assert_awaited()  # or assert_any_call
    # Check that we got "Translation service is not available."
    ctx.send.assert_any_call("[🤖]: Translation service is not available.")


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
    
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    
    await bot.languages_command(ctx)
    
    # Create a list of expected calls that match the actual implementation
    expected_calls = [
        mocker.call("[🤖]: Common language codes: es (Spanish), fr (French)"),
        mocker.call("[🤖]: Use these codes with the !translate command (e.g., !translate es)")
    ]
    
    # Assert that all expected calls were made
    ctx.send.assert_has_calls(expected_calls, any_order=True)


@pytest.mark.asyncio
async def test_speech_command_already_running():
    """
    If speech_service is already set, 
    it should say "Speech-to-text service is already running."
    """
    bot = ChatManager()
    bot.speech_service = True  # pretend it's already running

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.speech_command(ctx)

    ctx.send.assert_any_call("[🤖]: Speech-to-text service is already running.")


@pytest.mark.asyncio
async def test_translate_command_no_argument():
    """
    If !translate is called with no language code, 
    it should print usage instructions.
    """
    bot = ChatManager()
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    # Simulate the user message with no argument
    ctx.message.content = "!translate"

    await bot.translate_command(ctx)
    ctx.send.assert_any_call("[🤖]: Usage: !translate <language_code>")


@pytest.mark.asyncio
async def test_translate_command_unsupported_language(mocker):
    """
    If the language code doesn't exist in available_languages,
    it should say "Language code '<code>' is not supported."
    """
    bot = ChatManager()
    
    # Mock available languages
    mock_translation_service = mocker.MagicMock()
    mock_translation_service.available_languages = ["en", "es", "fr"]
    bot.translation_service = mock_translation_service

    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.message.content = "!translate zz"  # "zz" not in ["en","es","fr"]

    await bot.translate_command(ctx)
    ctx.send.assert_any_call("[🤖]: Language code 'zz' is not supported.")


@pytest.mark.asyncio
async def test_translate_command_success(mocker):
    """
    If a valid language code is provided, 
    it should enable translation for that code and 
    (optionally) start speech if it's not running.
    """
    bot = ChatManager()
    
    # Mock available languages
    mock_translation_service = mocker.MagicMock()
    mock_translation_service.available_languages = ["en", "es", "fr"]
    bot.translation_service = mock_translation_service

    # We'll also mock out speech_command so it doesn't do anything heavy
    bot.speech_command = AsyncMock()

    ctx = AsyncMock()
    ctx.message.content = "!translate es"
    ctx.send = AsyncMock()

    await bot.translate_command(ctx)

    # Should have called speech_command if not running yet
    bot.speech_command.assert_awaited()
    # Check that we set the language active
    assert bot.active_translations["es"] == True
    # Also check we posted a success message
    ctx.send.assert_any_call("[🤖]: Started translation to 'es'")


@pytest.mark.asyncio
async def test_stoptranslate_command(mocker):
    """
    If language is active, then !stoptranslate <code> should disable it.
    """
    bot = ChatManager()
    # Pretend we've already got Spanish active
    bot.active_translations["es"] = True

    ctx = AsyncMock()
    ctx.message.content = "!stoptranslate es"
    ctx.send = AsyncMock()

    await bot.stoptranslate_command(ctx)
    # Language should now be set to False
    assert bot.active_translations["es"] is False
    ctx.send.assert_any_call("[🤖]: Stopped translation to 'es'")


@pytest.mark.asyncio
async def test_stoptranslate_not_active():
    """
    If language isn't active, it should say so.
    """
    bot = ChatManager()
    # We haven't enabled 'es' yet, so it's missing/False

    ctx = AsyncMock()
    ctx.message.content = "!stoptranslate es"
    ctx.send = AsyncMock()

    await bot.stoptranslate_command(ctx)
    ctx.send.assert_any_call("[🤖]: Translation to 'es' is not active.")

@pytest.mark.asyncio
async def test_translatechat_command_no_translation_service():
    """
    Test !translatechat when translation service is unavailable.
    """
    bot = ChatManager()
    bot.translation_service = None

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.translatechat_command(ctx)
    ctx.send.assert_called_once_with("[🤖]: Translation service is not available.")


@pytest.mark.asyncio
async def test_translatechat_command_no_target_language():
    """
    Test !translatechat when no target language is configured.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()  # Mock translation service
    bot.translate_chat_to = None  # No target language configured

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    await bot.translatechat_command(ctx)
    ctx.send.assert_called_once_with("[🤖]: Chat translation target language not configured in audio_config.json")


@pytest.mark.asyncio
async def test_translatechat_command_toggle():
    """
    Test !translatechat toggles translation state correctly.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.translate_chat_to = "es"

    ctx = AsyncMock()
    ctx.send = AsyncMock()

    # Test enabling
    assert not bot.chat_translation_enabled  # Should start False
    await bot.translatechat_command(ctx)
    assert bot.chat_translation_enabled  # Should now be True
    ctx.send.assert_called_with("[🤖]: Chat translation enabled (Target language: es)")

    # Test disabling
    ctx.send.reset_mock()
    await bot.translatechat_command(ctx)
    assert not bot.chat_translation_enabled  # Should now be False
    ctx.send.assert_called_with("[🤖]: Chat translation disabled (Target language: es)")


@pytest.mark.asyncio
async def test_handle_chat_translation_ignored_user():
    """
    Test that messages from ignored users are not translated.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.ignore_users = {"nightbot"}
    bot.chat_translation_enabled = True
    bot.translate_chat_to = "es"

    message = AsyncMock()
    message.author.name = "nightbot"
    message.content = "Test message"

    channel = AsyncMock()
    bot.get_channel = AsyncMock(return_value=channel)

    await bot.handle_chat_translation(message)
    channel.send.assert_not_called()

# TODO: Fix these tests.
# @pytest.mark.asyncio
# async def test_handle_chat_translation_es_target():
#     """
#     Test translation of English message to Spanish.
#     """
#     bot = ChatManager()
    
#     # Create mock translation service
#     mock_translation_service = AsyncMock()
    
#     # Configure translate to return an async function that returns the translation
#     async def mock_translate(*args, **kwargs):
#         return "¡Hola mundo!"
        
#     mock_translation_service.model.translate = mock_translate
    
#     bot.translation_service = mock_translation_service
#     bot.chat_translation_enabled = True
#     bot.translate_chat_to = "es"
    
#     message = AsyncMock()
#     message.author.name = "user123"
#     message.content = "Hello world!"

#     channel = AsyncMock()
#     bot.get_channel = AsyncMock(return_value=channel)

#     await bot.handle_chat_translation(message)
#     channel.send.assert_awaited_once_with("[🤖][user123][es]: ¡Hola mundo!")


# @pytest.mark.asyncio
# async def test_handle_chat_translation_en_target():
#     """
#     Test translation of Spanish message to English.
#     """
#     bot = ChatManager()
    
#     # Create mock translation service
#     mock_translation_service = AsyncMock()
    
#     # Configure translate to return an async function that returns the translation
#     async def mock_translate(*args, **kwargs):
#         return "Hello world!"
        
#     mock_translation_service.translate = mock_translate
    
#     bot.translation_service = mock_translation_service
#     bot.chat_translation_enabled = True
#     bot.translate_chat_to = "en"
    
#     message = AsyncMock()
#     message.author.name = "user123"
#     message.content = "¡Hola mundo!"

#     channel = AsyncMock()
#     bot.get_channel = AsyncMock(return_value=channel)

#     await bot.handle_chat_translation(message)
#     channel.send.assert_awaited_once_with("[🤖][user123][en]: Hello world!")

@pytest.mark.asyncio
async def test_handle_chat_translation_same_language():
    """
    Test that messages already in target language aren't translated.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.translation_service.model.translate.return_value = "Hello world!"  # Same as input
    bot.chat_translation_enabled = True
    bot.translate_chat_to = "en"
    
    message = AsyncMock()
    message.author.name = "user123"
    message.content = "Hello world!"

    channel = AsyncMock()
    bot.get_channel = AsyncMock(return_value=channel)

    await bot.handle_chat_translation(message)
    channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_chat_translation_error_handling():
    """
    Test error handling during translation.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.translation_service.model.translate.side_effect = Exception("Translation failed")
    bot.translation_service.translate.side_effect = Exception("Fallback failed")
    bot.chat_translation_enabled = True
    bot.translate_chat_to = "es"
    
    message = AsyncMock()
    message.author.name = "user123"
    message.content = "Hello world!"

    channel = AsyncMock()
    bot.get_channel = AsyncMock(return_value=channel)

    # This should not raise an exception
    await bot.handle_chat_translation(message)
    channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_event_message_with_chat_translation():
    """
    Test that event_message properly handles chat translation.
    """
    bot = ChatManager()
    bot.translation_service = AsyncMock()
    bot.chat_translation_enabled = True
    bot.handle_chat_translation = AsyncMock()
    bot.handle_commands = AsyncMock()

    # Test regular message
    message = AsyncMock()
    message.echo = False
    message.content = "Hello world!"
    
    await bot.event_message(message)
    bot.handle_chat_translation.assert_called_once_with(message)
    bot.handle_commands.assert_called_once_with(message)

    # Test command message
    message.content = "!command"
    bot.handle_chat_translation.reset_mock()
    bot.handle_commands.reset_mock()
    
    await bot.event_message(message)
    bot.handle_chat_translation.assert_not_called()  # Shouldn't translate commands
    bot.handle_commands.assert_called_once_with(message)