"""Tests for Project Manager algorithm."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pm_algorithm import classify_task, assign_task, update_status


class TestPMAlgorithm:
    """Test cases for PM algorithm functions."""

    def test_classify_task_coding(self):
        """Test task classification for coding tasks."""
        coding_task = "Create a new HTML page with interactive elements"
        with patch('pm_algorithm.model') as mock_model, \
             patch('pm_algorithm.tokenizer') as mock_tokenizer:
            
            # Mock the tokenizer and model
            mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
            mock_outputs = MagicMock()
            mock_outputs.logits = [[0.8, 0.2]]  # Higher score for coding (index 0)
            mock_model.return_value = mock_outputs
            
            with patch('torch.no_grad'):
                with patch('torch.argmax', return_value=MagicMock(item=lambda: 0)):
                    result = classify_task(coding_task)
                    assert result == "coding"

    def test_classify_task_research(self):
        """Test task classification for research tasks."""
        research_task = "Research HTML5 best practices and modern web standards"
        with patch('pm_algorithm.model') as mock_model, \
             patch('pm_algorithm.tokenizer') as mock_tokenizer:
            
            # Mock the tokenizer and model
            mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
            mock_outputs = MagicMock()
            mock_outputs.logits = [[0.2, 0.8]]  # Higher score for research (index 1)
            mock_model.return_value = mock_outputs
            
            with patch('torch.no_grad'):
                with patch('torch.argmax', return_value=MagicMock(item=lambda: 1)):
                    result = classify_task(research_task)
                    assert result == "research"

    def test_classify_task_error_handling(self):
        """Test error handling in task classification."""
        with patch('pm_algorithm.tokenizer', side_effect=Exception("Tokenizer error")):
            with patch('pm_algorithm.log_error') as mock_log_error:
                result = classify_task("test task")
                assert result is None
                mock_log_error.assert_called_once()

    @patch('pm_algorithm.generate_code')
    @patch('pm_algorithm.send_feedback')
    @patch('pm_algorithm.update_status')
    @patch('pm_algorithm.log_success')
    def test_assign_task_coding(self, mock_log_success, mock_update_status, 
                               mock_send_feedback, mock_generate_code):
        """Test task assignment for coding tasks."""
        mock_generate_code.return_value = ("<html></html>", "Test passed")
        
        task_id = assign_task("coding", "Create a webpage")
        
        assert task_id is not None
        mock_generate_code.assert_called_once_with("Create a webpage")
        mock_send_feedback.assert_called_once_with("Test passed")
        mock_update_status.assert_called_once()
        mock_log_success.assert_called_once()

    @patch('pm_algorithm.generate_queries')
    @patch('pm_algorithm.store_results')
    @patch('pm_algorithm.update_status')
    @patch('pm_algorithm.log_success')
    def test_assign_task_research(self, mock_log_success, mock_update_status,
                                 mock_store_results, mock_generate_queries):
        """Test task assignment for research tasks."""
        mock_generate_queries.return_value = "Research summary"
        
        task_id = assign_task("research", "Research web frameworks")
        
        assert task_id is not None
        mock_generate_queries.assert_called_once_with("Research web frameworks")
        mock_store_results.assert_called_once_with("Research summary")
        mock_update_status.assert_called_once()
        mock_log_success.assert_called_once()

    def test_assign_task_unknown_category(self):
        """Test error handling for unknown task category."""
        with patch('pm_algorithm.log_error') as mock_log_error:
            task_id = assign_task("unknown", "Some task")
            assert task_id is None
            mock_log_error.assert_called_once()

    @patch('pm_algorithm.update_task_status')
    def test_update_status_success(self, mock_update_task_status):
        """Test successful status update."""
        update_status("test-task-id", "completed")
        mock_update_task_status.assert_called_once_with("test-task-id", "completed")

    @patch('pm_algorithm.update_task_status', side_effect=Exception("Update failed"))
    @patch('pm_algorithm.log_error')
    def test_update_status_error(self, mock_log_error, mock_update_task_status):
        """Test error handling in status update."""
        update_status("test-task-id", "completed")
        mock_log_error.assert_called_once()