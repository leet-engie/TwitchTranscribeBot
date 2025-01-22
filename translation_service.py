from typing import Optional, List
from easynmt import EasyNMT
import logging
import nltk
from config_manager import ConfigManager

class TranslationService:
    """
    A configurable service class for handling translations using EasyNMT.
    """
    
    def __init__(self, config_path: str = "audio_config.json"):
        """
        Initialize the translation service using configuration from JSON file.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize NLTK requirements
        self._initialize_nltk()
        
        # Load configuration using external ConfigManager
        self.config_manager = ConfigManager(config_path)
        self.translation_config = self.config_manager.get('translation', {
            'model': 'opus-mt',  # Default model if not specified
            'fallback_model': 'm2m100',
            'cache_dir': None
        })
        
        # Initialize primary model
        self.model_name = self.translation_config.get('model', 'opus-mt')
        self.cache_dir = self.translation_config.get('cache_dir')
        
        try:
            self.logger.info(f"Initializing primary model: {self.model_name}")
            self.model = EasyNMT(self.model_name, cache_dir=self.cache_dir)
            self.available_languages = self.model.get_languages()
        except Exception as e:
            self.logger.error(f"Failed to initialize primary model: {str(e)}")
            
            # Try fallback model if specified
            fallback_model = self.translation_config.get('fallback_model')
            if fallback_model:
                self.logger.info(f"Attempting fallback model: {fallback_model}")
                try:
                    self.model = EasyNMT(fallback_model, cache_dir=self.cache_dir)
                    self.model_name = fallback_model
                    self.available_languages = self.model.get_languages()
                except Exception as fallback_error:
                    raise RuntimeError(f"Both primary and fallback models failed. Last error: {str(fallback_error)}")
            else:
                raise

    @staticmethod
    def _initialize_nltk():
        """Initialize required NLTK data."""
        try:
            # Download required NLTK data
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {str(e)}")
            raise
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
            
        Raises:
            ValueError: If languages are not supported
            RuntimeError: If translation fails
        """
        # Validate languages
        if source_lang not in self.available_languages:
            raise ValueError(f"Source language '{source_lang}' is not supported")
        if target_lang not in self.available_languages:
            raise ValueError(f"Target language '{target_lang}' is not supported")
        
        try:
            return self.model.translate(text, source_lang=source_lang, target_lang=target_lang)
        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    @property
    def current_model(self) -> str:
        """Get the name of the currently active model."""
        return self.model_name


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the translation service
    try:
        translator = TranslationService()
        
        # Example translation
        text = "Hello, how are you?"
        translated = translator.translate(text, "en", "es")
        print(f"\nUsing model: {translator.current_model}")
        print(f"Original: {text}")
        print(f"Translated: {translated}")
    except Exception as e:
        print(f"Error during translation: {e}")