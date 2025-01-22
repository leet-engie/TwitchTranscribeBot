import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from command_handler import CommandHandler

@pytest.fixture
def mock_bot():
    bot = Mock()
    bot.bot_prefix = "!"
    bot.translation_service = Mock()
    bot.translation_service.available_languages = {'es', 'fr', 'de', 'en'}
    bot.speech_service = None
    bot.translate_chat_to = 'en'
    bot.chat_translation_enabled = False
    bot.ignore_users = set()
    return bot

@pytest.fixture
def handler(mock_bot):
    return CommandHandler(mock_bot)

@pytest.mark.asyncio
async def test_send_welcome_message(handler, mock_bot):
    mock_channel = AsyncMock()
    mock_bot.get_channel.return_value = mock_channel
    
    await handler.send_welcome_message()
    
    assert mock_channel.send.call_count == 7
    welcome_calls = mock_channel.send.call_args_list
    assert "Chat manager connected" in welcome_calls[0].args[0]
    assert "Available commands" in welcome_calls[1].args[0]

@pytest.mark.asyncio
async def test_handle_chat_translation_ignored_user(handler, mock_bot):
    mock_bot.ignore_users = {'ignored_user'}
    message = Mock()
    message.author.name = 'ignored_user'
    
    await handler.handle_chat_translation(message)
    
    mock_bot.get_channel.assert_not_called()

@pytest.mark.asyncio
async def test_handle_chat_translation_success(handler, mock_bot):
    message = Mock()
    message.author.name = 'test_user'
    message.content = 'Hola'
    
    mock_channel = AsyncMock()
    mock_bot.get_channel.return_value = mock_channel
    mock_bot.translation_service.translate.return_value = 'Hello'
    mock_bot.chat_translation_enabled = True
    
    await handler.handle_chat_translation(message)
    
    mock_channel.send.assert_called_once()
    assert 'Hello' in mock_channel.send.call_args.args[0]

@pytest.mark.asyncio
async def test_handle_languages_command(handler, mock_bot):
    ctx = AsyncMock()
    
    await handler.handle_languages_command(ctx)
    
    assert ctx.send.call_count == 2
    assert 'Common language codes' in ctx.send.call_args_list[0].args[0]

@pytest.mark.asyncio
async def test_handle_translatechat_command_toggle(handler, mock_bot):
    ctx = AsyncMock()
    initial_state = mock_bot.chat_translation_enabled
    
    await handler.handle_translatechat_command(ctx)
    
    assert mock_bot.chat_translation_enabled != initial_state
    ctx.send.assert_called_once()
    status = "enabled" if mock_bot.chat_translation_enabled else "disabled"
    assert status in ctx.send.call_args.args[0]

@pytest.mark.asyncio
async def test_handle_translate_command_invalid_language(handler, mock_bot):
    ctx = AsyncMock()
    ctx.message.content = '!translate invalid_lang'
    
    await handler.handle_translate_command(ctx)
    
    assert ctx.send.call_count == 2
    assert "not supported" in ctx.send.call_args_list[0].args[0]

@pytest.mark.asyncio
async def test_handle_translate_command_success(handler, mock_bot):
    ctx = AsyncMock()
    ctx.message.content = '!translate es'
    mock_bot.speech_service = Mock()  # Prevent speech service initialization
    
    await handler.handle_translate_command(ctx)
    
    assert ctx.send.call_count == 1
    assert "Started translation to 'es'" in ctx.send.call_args.args[0]
    assert handler.active_translations['es'] is True

@pytest.mark.asyncio
async def test_handle_stoptranslate_command(handler, mock_bot):
    ctx = AsyncMock()
    ctx.message.content = '!stoptranslate es'
    handler.active_translations['es'] = True
    
    await handler.handle_stoptranslate_command(ctx)
    
    assert ctx.send.call_count == 1
    assert "Stopped translation" in ctx.send.call_args.args[0]
    assert not handler.active_translations['es']

@pytest.mark.asyncio
async def test_translate_message_fallback(handler, mock_bot):
    content = "Hello"
    mock_bot.translation_service.translate.side_effect = [Exception(), "Hola"]
    
    result = await handler._translate_message(content)
    
    assert result == "Hola"
    assert mock_bot.translation_service.translate.call_count == 2