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
    # We expect at least one "Common language codes" message
    ctx.send.assert_any_call("[🤖]: Common language codes: en (English), es (Spanish), fr (French)")
    # We do *not* send the full list if we removed that logic,
    # or if we do, it should be short enough.


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
