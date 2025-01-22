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