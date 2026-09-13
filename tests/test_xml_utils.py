"""Tests for XML utilities module."""

import pytest
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xml_utils import (
    get_xml_file_path, create_xml_schema, update_task_status,
    log_success, log_failure, log_error, store_results
)


class TestXMLUtils:
    """Test cases for XML utility functions."""

    def test_get_xml_file_path_default(self):
        """Test getting default XML file path."""
        with patch.dict(os.environ, {}, clear=True):
            path = get_xml_file_path()
            assert path == 'data/project_data.xml'

    def test_get_xml_file_path_custom(self):
        """Test getting custom XML file path from environment."""
        with patch.dict(os.environ, {'XML_DATA_PATH': 'custom/path/data.xml'}):
            path = get_xml_file_path()
            assert path == 'custom/path/data.xml'

    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    @patch('xml.etree.ElementTree.ElementTree.write')
    def test_create_xml_schema_new_file(self, mock_write, mock_exists, mock_makedirs):
        """Test creating new XML schema file."""
        create_xml_schema()
        mock_makedirs.assert_called_once()
        mock_write.assert_called_once()

    @patch('os.path.exists', return_value=True)
    def test_create_xml_schema_existing_file(self, mock_exists):
        """Test when XML schema file already exists."""
        # Should not raise any exception
        create_xml_schema()

    def test_update_task_status_integration(self):
        """Integration test for updating task status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml_path = os.path.join(temp_dir, 'test_data.xml')
            
            with patch('xml_utils.get_xml_file_path', return_value=test_xml_path):
                # Create initial XML structure
                create_xml_schema()
                
                # Update task status
                update_task_status('test-task-1', 'in_progress')
                
                # Verify the XML content
                tree = ET.parse(test_xml_path)
                root = tree.getroot()
                task = root.find(".//Task[TaskID='test-task-1']")
                assert task is not None
                status = task.find('Status')
                assert status is not None
                assert status.text == 'in_progress'

    def test_log_success_integration(self):
        """Integration test for logging success."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml_path = os.path.join(temp_dir, 'test_data.xml')
            
            with patch('xml_utils.get_xml_file_path', return_value=test_xml_path):
                # Create initial XML and task
                create_xml_schema()
                update_task_status('test-task-1', 'pending')
                
                # Log success
                log_success('test-task-1')
                
                # Verify the XML content
                tree = ET.parse(test_xml_path)
                root = tree.getroot()
                task = root.find(".//Task[TaskID='test-task-1']")
                assert task is not None
                log_elem = task.find('Log')
                assert log_elem is not None
                assert log_elem.text == 'Success'

    def test_log_failure_integration(self):
        """Integration test for logging failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml_path = os.path.join(temp_dir, 'test_data.xml')
            
            with patch('xml_utils.get_xml_file_path', return_value=test_xml_path):
                # Create initial XML and task
                create_xml_schema()
                update_task_status('test-task-1', 'pending')
                
                # Log failure
                error_message = "Test error occurred"
                log_failure('test-task-1', error_message)
                
                # Verify the XML content
                tree = ET.parse(test_xml_path)
                root = tree.getroot()
                task = root.find(".//Task[TaskID='test-task-1']")
                assert task is not None
                log_elem = task.find('Log')
                assert log_elem is not None
                assert f"Failure: {error_message}" in log_elem.text

    def test_log_error_integration(self):
        """Integration test for logging general errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml_path = os.path.join(temp_dir, 'test_data.xml')
            
            with patch('xml_utils.get_xml_file_path', return_value=test_xml_path):
                # Log error
                error_message = "General error occurred"
                log_error(error_message)
                
                # Verify the XML content
                tree = ET.parse(test_xml_path)
                root = tree.getroot()
                errors = root.find('Errors')
                assert errors is not None
                error_elem = errors.find('Error')
                assert error_elem is not None
                assert error_elem.text == error_message
                
                # Check timestamp exists
                timestamp = error_elem.find('Timestamp')
                assert timestamp is not None
                assert timestamp.text is not None

    def test_store_results_integration(self):
        """Integration test for storing research results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml_path = os.path.join(temp_dir, 'test_data.xml')
            
            with patch('xml_utils.get_xml_file_path', return_value=test_xml_path):
                # Store results
                summary = "Research summary with findings"
                store_results(summary)
                
                # Verify the XML content
                tree = ET.parse(test_xml_path)
                root = tree.getroot()
                results = root.find('Results')
                assert results is not None
                result_elem = results.find('Result')
                assert result_elem is not None
                assert result_elem.text == summary
                
                # Check timestamp exists
                timestamp = result_elem.find('Timestamp')
                assert timestamp is not None
                assert timestamp.text is not None