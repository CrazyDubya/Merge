"""Tests for utilities module."""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_configuration, substitute_env_variables, validate_api_keys, setup_logging


class TestUtils:
    """Test cases for utility functions."""

    def test_substitute_env_variables_simple(self):
        """Test simple environment variable substitution."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            content = "api_key: ${TEST_VAR}"
            result = substitute_env_variables(content)
            assert result == "api_key: test_value"

    def test_substitute_env_variables_with_default(self):
        """Test environment variable substitution with default value."""
        content = "api_key: ${MISSING_VAR:-default_value}"
        result = substitute_env_variables(content)
        assert result == "api_key: default_value"

    def test_substitute_env_variables_existing_with_default(self):
        """Test environment variable substitution when var exists with default."""
        with patch.dict(os.environ, {'EXISTING_VAR': 'actual_value'}):
            content = "api_key: ${EXISTING_VAR:-default_value}"
            result = substitute_env_variables(content)
            assert result == "api_key: actual_value"

    def test_validate_api_keys_valid(self):
        """Test API key validation with valid key."""
        config = {'anthropic_api_key': 'valid_key_123'}
        # Should not raise an exception
        validate_api_keys(config)

    def test_validate_api_keys_invalid(self):
        """Test API key validation with invalid keys."""
        config = {
            'anthropic_api_key': 'your_anthropic_api_key_here',
            'openai_api_key': '',
            'deepseek_api_key': None
        }
        with pytest.raises(ValueError, match="At least one valid API key"):
            validate_api_keys(config)

    def test_validate_api_keys_empty(self):
        """Test API key validation with empty config."""
        config = {}
        with pytest.raises(ValueError, match="At least one valid API key"):
            validate_api_keys(config)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="""
anthropic_api_key: ${ANTHROPIC_API_KEY}
openai_api_key: ${OPENAI_API_KEY:-}
app:
  name: "Test App"
logging:
  level: ${LOG_LEVEL:-INFO}
""")
    def test_load_configuration_success(self, mock_file, mock_exists):
        """Test successful configuration loading."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key_123',
            'LOG_LEVEL': 'DEBUG'
        }):
            config = load_configuration()
            assert config['anthropic_api_key'] == 'test_key_123'
            assert config['app']['name'] == 'Test App'
            assert config['logging']['level'] == 'DEBUG'

    @patch('pathlib.Path.exists', return_value=False)
    def test_load_configuration_file_not_found(self, mock_exists):
        """Test configuration loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_configuration()

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', side_effect=IOError("File read error"))
    def test_load_configuration_read_error(self, mock_file, mock_exists):
        """Test configuration loading with file read error."""
        with pytest.raises(Exception, match="Error loading configuration"):
            load_configuration()

    @patch('os.makedirs')
    @patch('logging.basicConfig')
    def test_setup_logging_default(self, mock_basic_config, mock_makedirs):
        """Test logging setup with default configuration."""
        logger = setup_logging()
        assert logger is not None
        mock_makedirs.assert_called_once()
        mock_basic_config.assert_called_once()

    @patch('os.makedirs')
    @patch('logging.basicConfig')
    def test_setup_logging_custom_config(self, mock_basic_config, mock_makedirs):
        """Test logging setup with custom configuration."""
        config = {
            'logging': {
                'level': 'DEBUG',
                'file': 'custom/app.log'
            }
        }
        logger = setup_logging(config)
        assert logger is not None
        mock_makedirs.assert_called_once_with('custom', exist_ok=True)