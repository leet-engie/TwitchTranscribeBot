# config_manager.py
import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file: str = 'audio_config.json'):
        """Initialize the configuration manager."""
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self.config = {
                    'chat_translation': {
                        'translate_to': 'en',
                        'ignore_users': []
                    }
                }
                # Create default config file
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}

    def save_config(self):
        """Save current configuration to JSON file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
        self.save_config()

# translation_service.py
import logging
from typing import Optional
import requests
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        """Initialize the translation service."""
        self.available_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French',
            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'nl': 'Dutch', 'pl': 'Polish', 'ru': 'Russian',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'
        }
        
        # Initialize the model and tokenizer (using Helsinki-NLP's MarianMT)
        try:
            model_name = 'Helsinki-NLP/opus-mt-es-en'  # Example model
            self.model = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language.
        Returns None if translation fails.
        """
        if not text or not source_lang or not target_lang:
            return None

        try:
            # Tokenize and translate
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            translated = self.model.generate(**inputs)
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return result
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language code is supported."""
        return lang_code.lower() in self.available_languages