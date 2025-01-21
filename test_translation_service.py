# test_translation_service.py
import pytest
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from translation_service import TranslationService, ConfigManager

# Test data
VALID_CONFIG = {
    "audio_device_index": 13,
    "sample_rate": 16000,
    "translation": {
        "model": "opus-mt",
        "fallback_model": "m2m100",
        "cache_dir": None
    }
}

VALID_CONFIG_NO_TRANSLATION = {
    "audio_device_index": 13,
    "sample_rate": 16000
}

@pytest.fixture
def mock_easynmt():
    with patch('translation_service.EasyNMT') as mock:
        # Configure the mock
        instance = mock.return_value
        instance.get_languages.return_value = ['en', 'es', 'fr']
        instance.translate.return_value = "¡Hola, ¿cómo estás?"
        yield mock

@pytest.fixture
def mock_nltk():
    with patch('translation_service.nltk') as mock:
        yield mock

@pytest.fixture
def config_file(tmp_path):
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(VALID_CONFIG, f)
    return config_path

class TestConfigManager:
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file"""
        config_path = tmp_path / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(VALID_CONFIG, f)
        
        manager = ConfigManager(str(config_path))
        assert manager.config == VALID_CONFIG
        
    def test_load_config_with_no_translation(self, tmp_path):
        """Test loading config without translation section"""
        config_path = tmp_path / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(VALID_CONFIG_NO_TRANSLATION, f)
        
        manager = ConfigManager(str(config_path))
        translation_config = manager.get_translation_config()
        assert translation_config['model'] == 'opus-mt'
        assert translation_config['fallback_model'] == 'm2m100'
        
    def test_load_nonexistent_config(self):
        """Test attempting to load a non-existent config file"""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent.json")
            
    def test_load_invalid_json(self, tmp_path):
        """Test loading an invalid JSON file"""
        config_path = tmp_path / "invalid_config.json"
        with open(config_path, 'w') as f:
            f.write("invalid json content")
            
        with pytest.raises(ValueError):
            ConfigManager(str(config_path))

class TestTranslationService:
    def test_initialization(self, mock_easynmt, mock_nltk, config_file):
        """Test successful initialization of translation service"""
        service = TranslationService(str(config_file))
        assert service.current_model == 'opus-mt'
        assert mock_nltk.download.call_count == 3  # punkt, punkt_tab, averaged_perceptron_tagger
        
    def test_fallback_model(self, mock_easynmt, mock_nltk, config_file):
        """Test fallback to secondary model when primary fails"""
        # Make primary model fail
        mock_easynmt.side_effect = [Exception("Primary model failed"), Mock()]
        
        service = TranslationService(str(config_file))
        assert service.current_model == 'm2m100'
        
    def test_both_models_fail(self, mock_easynmt, mock_nltk, config_file):
        """Test behavior when both primary and fallback models fail"""
        # Make both models fail
        mock_easynmt.side_effect = [
            Exception("Primary model failed"),
            Exception("Fallback model failed")
        ]
        
        with pytest.raises(RuntimeError) as exc_info:
            TranslationService(str(config_file))
        assert "Both primary and fallback models failed" in str(exc_info.value)
        
    def test_nltk_initialization_failure(self, mock_easynmt, mock_nltk, config_file):
        """Test handling of NLTK initialization failure"""
        mock_nltk.download.side_effect = Exception("NLTK download failed")
        
        with pytest.raises(Exception) as exc_info:
            TranslationService(str(config_file))
        assert "NLTK download failed" in str(exc_info.value)
        
    def test_successful_translation(self, mock_easynmt, mock_nltk, config_file):
        """Test successful translation"""
        service = TranslationService(str(config_file))
        result = service.translate("Hello, how are you?", "en", "es")
        assert result == "¡Hola, ¿cómo estás?"
        
    def test_unsupported_source_language(self, mock_easynmt, mock_nltk, config_file):
        """Test translation with unsupported source language"""
        service = TranslationService(str(config_file))
        with pytest.raises(ValueError) as exc_info:
            service.translate("Hello", "xx", "es")
        assert "Source language 'xx' is not supported" in str(exc_info.value)
        
    def test_unsupported_target_language(self, mock_easynmt, mock_nltk, config_file):
        """Test translation with unsupported target language"""
        service = TranslationService(str(config_file))
        with pytest.raises(ValueError) as exc_info:
            service.translate("Hello", "en", "xx")
        assert "Target language 'xx' is not supported" in str(exc_info.value)
        
    def test_translation_failure(self, mock_easynmt, mock_nltk, config_file):
        """Test handling of translation failure"""
        service = TranslationService(str(config_file))
        mock_easynmt.return_value.translate.side_effect = Exception("Translation failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            service.translate("Hello", "en", "es")
        assert "Translation failed" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main(["-v", "--cov=translation_service", "test_translation_service.py"])