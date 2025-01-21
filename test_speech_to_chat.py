import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pyaudio
import webrtcvad
import os
from pathlib import Path

# Import the classes to test
from speech_to_chat import AudioTranscriber, Bot

# Fixtures
@pytest.fixture
def mock_config():
    return {
        "audio_device_index": 2,
        "sample_rate": 16000,
        "chunk_size": 480,
        "channels": 1,
        "filtered_phrases": [
            "Thanks for watching!",
            "Like and subscribe"
        ]
    }

@pytest.fixture
def mock_bot():
    bot = Mock()
    bot.bot_prefix = "[🎙️]: "
    bot.get_channel = Mock(return_value=AsyncMock())
    return bot

@pytest.fixture
def mock_pyaudio():
    with patch('pyaudio.PyAudio') as mock:
        # Create a more realistic device info mock
        def get_device_info_by_index(index):
            devices = {
                0: {'maxInputChannels': 2, 'name': 'Default Input Device'},
                1: {'maxInputChannels': 0, 'name': 'Output Only Device'},
                2: {'maxInputChannels': 2, 'name': 'Test Microphone'}
            }
            return devices.get(index, {'maxInputChannels': 0, 'name': 'Unknown'})
        
        mock.return_value.get_device_info_by_index.side_effect = get_device_info_by_index
        mock.return_value.get_device_count.return_value = 3
        
        # Mock audio stream
        stream_mock = Mock()
        stream_mock.read.return_value = np.random.bytes(960)  # 480 samples * 2 bytes
        mock.return_value.open.return_value = stream_mock
        
        yield mock

@pytest.fixture
def mock_whisper():
    with patch('whisper.load_model') as mock:
        model_mock = Mock()
        model_mock.transcribe.return_value = {"text": "Test transcription"}
        mock.return_value = model_mock
        yield mock

@pytest.fixture
def mock_vad():
    with patch('webrtcvad.Vad') as mock:
        vad_instance = Mock()
        vad_instance.is_speech.return_value = True
        mock.return_value = vad_instance
        yield mock

# Tests for AudioTranscriber
class TestAudioTranscriber:
    @pytest.mark.asyncio
    async def test_init(self, mock_bot, mock_config, mock_pyaudio, mock_whisper, mock_vad):
        """Test AudioTranscriber initialization"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        assert transcriber.bot == mock_bot
        assert transcriber.config == mock_config
        assert transcriber.is_recording == True
        assert transcriber.model_name == 'medium'

    def test_validate_audio_device(self, mock_bot, mock_config, mock_pyaudio):
        """Test audio device validation"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        transcriber.validate_audio_device()
        assert mock_pyaudio.return_value.get_device_info_by_index.called

    def test_should_filter_phrase(self, mock_bot, mock_config, mock_pyaudio, mock_whisper):
        """Test phrase filtering"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        
        # Test phrase that should be filtered
        should_filter, phrase = transcriber.should_filter_phrase("Thanks for watching!")
        assert should_filter == True
        assert phrase == "Thanks for watching!"
        
        # Test phrase that should not be filtered
        should_filter, phrase = transcriber.should_filter_phrase("Hello world")
        assert should_filter == False
        assert phrase is None

    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, mock_bot, mock_config, mock_pyaudio, mock_whisper):
        """Test audio chunk processing"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        
        # Create test audio data
        audio_data = np.random.randn(16000).astype(np.int16).tobytes()
        
        # Process the audio chunk
        await transcriber.process_audio_chunk(audio_data)
        
        # Verify that the model was called
        transcriber.model.transcribe.assert_called_once()
        
        # Verify that the bot tried to send a message
        channel = mock_bot.get_channel.return_value
        channel.send.assert_called_once_with(f"{mock_bot.bot_prefix}Test transcription")

    @pytest.mark.asyncio
    async def test_process_audio_chunk_with_sample(self, mock_bot, mock_config, mock_pyaudio, mock_whisper):
        """Test audio processing with a pre-recorded sample"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        
        # Create a simple sine wave as test audio (1 second at 440Hz)
        duration = 1  # seconds
        sample_rate = mock_config['sample_rate']
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Configure Whisper mock to return a specific transcription
        mock_transcription = "Test transcription from audio sample"
        transcriber.model.transcribe.return_value = {"text": mock_transcription}
        
        # Process the audio chunk
        await transcriber.process_audio_chunk(audio_data.tobytes())
        
        # Verify Whisper model was called with correct data type
        model_call_args = transcriber.model.transcribe.call_args[0][0]
        assert isinstance(model_call_args, np.ndarray), "Model should receive numpy array"
        assert model_call_args.dtype == np.float32, "Model should receive float32 data"
        assert not np.any(np.abs(model_call_args) > 1.0), "Audio should be normalized between -1 and 1"
        
        # Verify the bot sent the transcribed message
        channel = mock_bot.get_channel.return_value
        expected_message = f"{mock_bot.bot_prefix}{mock_transcription}"
        channel.send.assert_called_once_with(expected_message)

    def test_stop(self, mock_bot, mock_config, mock_pyaudio):
        """Test stopping the transcriber"""
        transcriber = AudioTranscriber(mock_bot, mock_config)
        transcriber.stop()
        
        assert transcriber.is_recording == False
        assert transcriber.stream.stop_stream.called
        assert transcriber.stream.close.called
        assert transcriber.p.terminate.called

# Tests for Bot
class TestBot:
    @pytest.fixture
    def mock_env_vars(self):
        with patch.dict(os.environ, {
            'TMI_TOKEN': 'test_token',
            'BOT_NICK': 'test_bot',
            'CHANNEL': 'test_channel'
        }):
            yield

    def test_bot_init(self, mock_env_vars):
        """Test Bot initialization"""
        with patch('json.load') as mock_json:
            mock_json.return_value = {
                "audio_device_index": 0,
                "filtered_phrases": ["Thanks for watching!"]
            }
            
            bot = Bot()
            assert bot.bot_prefix == "[🎙️]: "
            assert bot.should_stop == False
            assert bot.transcriber is None

    @pytest.mark.asyncio
    async def test_event_ready(self, mock_env_vars):
        """Test bot ready event"""
        with patch('twitchio.ext.commands.Bot.__init__', return_value=None), \
             patch('json.load', return_value={"audio_device_index": 0}), \
             patch('speech_to_chat.AudioTranscriber') as MockTranscriber:

            # Create bot instance with mocked internal TwitchIO attributes
            bot = Bot()
            
            # Create mock http and connection objects with nick property
            mock_http = MagicMock()
            mock_http.nick = 'test_bot'
            bot._http = mock_http
            
            mock_connection = MagicMock()
            mock_connection.nick = 'test_bot'
            bot._connection = mock_connection
            
            # Setup mock channel
            mock_channel = AsyncMock()
            mock_channel.send = AsyncMock()
            bot.get_channel = Mock(return_value=mock_channel)
            
            # Setup mock transcriber
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.record_and_transcribe = AsyncMock()
            MockTranscriber.return_value = mock_transcriber_instance
            
            # Call event_ready
            await bot.event_ready()
            
            # Verify channel message was sent
            mock_channel.send.assert_called_once_with("[🎙️]: Connected and listening!")
            
            # Verify transcriber was created and started
            MockTranscriber.assert_called_once()
            assert bot.transcriber is not None
            assert hasattr(bot, 'transcribe_task')

    def test_cleanup(self, mock_env_vars):
        """Test bot cleanup"""
        with patch('json.load'):
            bot = Bot()
            bot.transcribe_task = Mock()
            bot.transcriber = Mock()
            
            bot.cleanup()
            
            assert bot.transcribe_task.cancel.called
            assert bot.transcriber.stop.called

if __name__ == '__main__':
    pytest.main(['-v'])