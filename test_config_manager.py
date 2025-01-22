import pytest
import json
import os
from config_manager import ConfigManager

@pytest.fixture
def config_file():
    """Temporary config file for testing."""
    filename = "test_config.json"
    yield filename
    # Cleanup after tests
    if os.path.exists(filename):
        os.remove(filename)

@pytest.fixture
def config_manager(config_file):
    """ConfigManager instance for testing."""
    return ConfigManager(config_file)

def test_init_with_nonexistent_file(config_file):
    """Test initialization with a non-existent config file."""
    manager = ConfigManager(config_file)
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    assert config == {
        'chat_translation': {
            'translate_to': 'en',
            'ignore_users': []
        }
    }

def test_init_with_existing_file(config_file):
    """Test initialization with an existing config file."""
    test_config = {'test_key': 'test_value'}
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    manager = ConfigManager(config_file)
    assert manager.config == test_config

def test_load_config_with_invalid_json(config_file):
    """Test loading invalid JSON content."""
    with open(config_file, 'w') as f:
        f.write('invalid json content')
    
    manager = ConfigManager(config_file)
    assert manager.config == {}

def test_save_config(config_manager):
    """Test saving configuration to file."""
    test_config = {'new_key': 'new_value'}
    config_manager.config = test_config
    config_manager.save_config()
    
    with open(config_manager.config_file, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == test_config

def test_get_existing_key(config_manager):
    """Test getting an existing configuration value."""
    config_manager.config = {'test_key': 'test_value'}
    assert config_manager.get('test_key') == 'test_value'

def test_get_nonexistent_key(config_manager):
    """Test getting a non-existent configuration value."""
    assert config_manager.get('nonexistent_key') is None
    assert config_manager.get('nonexistent_key', 'default') == 'default'

def test_set_new_key(config_manager):
    """Test setting a new configuration value."""
    config_manager.set('new_key', 'new_value')
    assert config_manager.config['new_key'] == 'new_value'
    
    # Verify it was saved to file
    with open(config_manager.config_file, 'r') as f:
        saved_config = json.load(f)
    assert saved_config['new_key'] == 'new_value'

def test_set_existing_key(config_manager):
    """Test updating an existing configuration value."""
    config_manager.config = {'existing_key': 'old_value'}
    config_manager.set('existing_key', 'new_value')
    assert config_manager.config['existing_key'] == 'new_value'

def test_save_config_with_permission_error(config_manager, monkeypatch):
    """Test saving config when file is not writable."""
    def mock_open(*args, **kwargs):
        raise PermissionError("Permission denied")
    
    monkeypatch.setattr('builtins.open', mock_open)
    config_manager.save_config()  # Should log error but not raise exception

def test_load_config_with_permission_error(config_file, monkeypatch):
    """Test loading config when file is not readable."""
    def mock_open(*args, **kwargs):
        raise PermissionError("Permission denied")
    
    def mock_exists(*args, **kwargs):
        return True  # Pretend file exists but can't be opened
    
    monkeypatch.setattr('builtins.open', mock_open)
    monkeypatch.setattr('os.path.exists', mock_exists)
    manager = ConfigManager(config_file)
    assert manager.config == {}