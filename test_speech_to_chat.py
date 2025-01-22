import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, mock_open

import numpy as np

from speech_to_chat import AudioTranscriber

@pytest.fixture
def mock_bot():
    """Fixture to create a mock bot with a send_transcription coroutine."""
    class MockBot:
        async def send_transcription(self, text):
            pass
    return MockBot()

@pytest.fixture
def mock_pyaudio():
    """Fixture to mock pyaudio and return MagicMock for PyAudio and its stream."""
    with patch("pyaudio.PyAudio") as mock_py:
        mock_py_instance = MagicMock()
        mock_stream = MagicMock()
        mock_py_instance.open.return_value = mock_stream
        mock_py.return_value = mock_py_instance
        yield mock_py_instance, mock_stream

@pytest.fixture
def mock_torch():
    """Fixture to mock torch.cuda functionality."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("torch.backends.cuda.matmul", create=True), \
         patch("torch.backends.cudnn", create=True):
        yield mock_empty_cache

@pytest.fixture
def mock_whisper_load_model():
    """Fixture to mock whisper.load_model."""
    with patch("whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test transcription"}
        mock_load.return_value = mock_model
        yield mock_load

@pytest.fixture
def mock_webrtcvad():
    """Fixture to mock webrtcvad.Vad."""
    with patch("webrtcvad.Vad") as mock_vad_class:
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = False
        mock_vad_class.return_value = mock_vad_instance
        yield mock_vad_instance

@pytest.fixture
def mock_config_manager():
    """Fixture to mock ConfigManager."""
    with patch("speech_to_chat.ConfigManager") as mock_cm:
        mock_cm_instance = MagicMock()
        # Mock the new nested configuration structure
        mock_cm_instance.get.return_value = {
            "device": {
                "audio_device_index": 1,
                "sample_rate": 16000,
                "chunk_size": 480,
                "channels": 1
            },
            "whisper": {
                "model_name": "medium"
            },
            "voice_detection": {
                "vad_mode": 2,
                "silence_threshold": 20
            },
            "filtering": {
                "filtered_phrases": ["secret"]
            }
        }
        mock_cm.return_value = mock_cm_instance
        yield mock_cm_instance

@pytest.mark.asyncio
class TestAudioTranscriber:
    @pytest.fixture
    def setup_transcriber(
        self, 
        mock_bot, 
        mock_pyaudio, 
        mock_torch, 
        mock_whisper_load_model, 
        mock_webrtcvad,
        mock_config_manager
    ):
        """Helper fixture to instantiate AudioTranscriber with all necessary mocks."""
        transcriber = AudioTranscriber(bot=mock_bot)
        return transcriber

    async def test_init_loads_config(self, setup_transcriber):
        """Test that the constructor loads config and sets expected defaults."""
        transcriber = setup_transcriber
        
        # Check that sample_rate and chunk_size are set to the expected values
        assert transcriber.audio_config['sample_rate'] == 16000
        assert transcriber.audio_config['chunk_size'] == 480
        assert transcriber.audio_config['channels'] == 1
        assert transcriber.audio_config['filtered_phrases'] == ["secret"]

    async def test_process_audio_chunk_small_data(self, setup_transcriber):
        """Test that process_audio_chunk doesn't transcribe when audio data is too short."""
        transcriber = setup_transcriber
        
        # Create a small buffer which is less than 0.5 seconds
        small_data = np.zeros(100, dtype=np.int16).tobytes()
        
        await transcriber.process_audio_chunk(small_data)
        transcriber.model.transcribe.assert_not_called()

    async def test_process_audio_chunk_enough_data(self, setup_transcriber, mock_bot):
        """Test that process_audio_chunk transcribes when audio data is large enough."""
        transcriber = setup_transcriber
        
        # Create 1 second of audio data
        big_data = np.zeros(16000, dtype=np.int16).tobytes()
        await transcriber.process_audio_chunk(big_data)
        
        # Verify the model transcribed the audio
        transcriber.model.transcribe.assert_called_once()

    def test_test_audio_levels(self, setup_transcriber):
        """Test the audio level testing functionality."""
        transcriber = setup_transcriber
        
        # Mock the stream.read to return consistent data
        transcriber.stream.read = MagicMock(
            return_value=np.ones(transcriber.audio_config['chunk_size'], dtype=np.int16).tobytes()
        )
        
        result = transcriber.test_audio_levels(duration=1)
        assert isinstance(result, bool)

    def test_stop(self, setup_transcriber):
        """Test that stop() correctly closes the audio stream and terminates PyAudio."""
        transcriber = setup_transcriber
        mock_stream = transcriber.stream
        mock_p = transcriber.p
        
        transcriber.stop()
        
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_p.terminate.assert_called_once()

    def test_filter_transcription(self, setup_transcriber):
        """Test that filtered phrases are removed from transcriptions."""
        transcriber = setup_transcriber
        
        # Test cases with filtered phrases
        test_cases = [
            ("This is a secret message", "This is a message"),
            ("This secret is hidden", "This is hidden"),
            ("A secret secret message", "A message"),
            ("No filtered words here", "No filtered words here"),
            ("  secret  ", ""),  # Test with spaces around word
            ("", ""),  # Test empty string
            (None, None),  # Test None input
        ]
        
        for input_text, expected_output in test_cases:
            filtered = transcriber.filter_transcription(input_text)
            if filtered is not None:
                filtered = ' '.join(filtered.split())  # Normalize spaces
            assert filtered == expected_output, f"Failed on input: '{input_text}'"