"""Tests for Coder algorithm."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coder_algorithm import generate_code, test_code, send_feedback


class TestCoderAlgorithm:
    """Test cases for Coder algorithm functions."""

    def test_generate_code_success(self):
        """Test successful code generation."""
        task_details = "Create a simple HTML page"
        
        with patch('coder_algorithm.test_code', return_value="Code tested successfully"):
            code, test_result = generate_code(task_details)
            
            assert "<html>" in code
            assert task_details in code
            assert test_result == "Code tested successfully"

    def test_generate_code_error(self):
        """Test error handling in code generation."""
        task_details = "Create a simple HTML page"
        
        with patch('coder_algorithm.test_code', side_effect=Exception("Test error")):
            code, test_result = generate_code(task_details)
            
            assert code == ""
            assert "Error generating code" in test_result

    @patch('builtins.open', new_callable=mock_open)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.run')
    def test_test_code_linux_success(self, mock_subprocess, mock_platform, mock_file):
        """Test successful code testing on Linux."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = test_code("<html><body>Test</body></html>")
        
        assert result == "Code tested successfully"
        mock_file.assert_called_once_with("temp_code.html", "w")
        mock_subprocess.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('platform.system', return_value='Darwin')
    @patch('subprocess.run')
    def test_test_code_macos_success(self, mock_subprocess, mock_platform, mock_file):
        """Test successful code testing on macOS."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = test_code("<html><body>Test</body></html>")
        
        assert result == "Code tested successfully"
        mock_subprocess.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('platform.system', return_value='Windows')
    @patch('subprocess.run')
    def test_test_code_windows_success(self, mock_subprocess, mock_platform, mock_file):
        """Test successful code testing on Windows."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = test_code("<html><body>Test</body></html>")
        
        assert result == "Code tested successfully"
        mock_subprocess.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('platform.system', return_value='Unknown')
    def test_test_code_unsupported_os(self, mock_platform, mock_file):
        """Test code testing on unsupported OS."""
        result = test_code("<html><body>Test</body></html>")
        
        assert result == "Unsupported operating system"

    @patch('builtins.open', new_callable=mock_open)
    @patch('platform.system', return_value='Linux')
    @patch('subprocess.run', side_effect=Exception("Command failed"))
    def test_test_code_subprocess_error(self, mock_subprocess, mock_platform, mock_file):
        """Test error handling in code testing."""
        result = test_code("<html><body>Test</body></html>")
        
        assert "Test failed" in result

    @patch('coder_algorithm.log_success')
    def test_send_feedback_success(self, mock_log_success):
        """Test feedback for successful test."""
        send_feedback("Code tested successfully")
        mock_log_success.assert_called_once()

    @patch('coder_algorithm.log_failure')
    def test_send_feedback_failure(self, mock_log_failure):
        """Test feedback for failed test."""
        send_feedback("Test failed: syntax error")
        mock_log_failure.assert_called_once_with("task_id", "Test failed: syntax error")