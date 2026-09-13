"""Tests for Researcher algorithm."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researcher_algorithm import generate_queries, fetch_data, summarize_results, consult_llm_for_queries


class TestResearcherAlgorithm:
    """Test cases for Researcher algorithm functions."""

    @patch('researcher_algorithm.consult_llm_for_queries')
    @patch('researcher_algorithm.fetch_data')
    @patch('researcher_algorithm.summarize_results')
    def test_generate_queries_success(self, mock_summarize, mock_fetch, mock_consult):
        """Test successful query generation and research."""
        task_details = "HTML5 best practices"
        mock_consult.return_value = ["query1", "query2", "enhanced1", "enhanced2"]
        mock_fetch.return_value = [{"title": "Test", "description": "Test desc"}]
        mock_summarize.return_value = "Research summary"
        
        result = generate_queries(task_details)
        
        assert result == "Research summary"
        mock_consult.assert_called_once()
        mock_fetch.assert_called_once()
        mock_summarize.assert_called_once()

    @patch('requests.get')
    @patch('researcher_algorithm.log_error')
    def test_fetch_data_success(self, mock_log_error, mock_requests):
        """Test successful data fetching."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.text = '''
        <div class="tF2Cxc">
            <h3>Test Title</h3>
            <div class="VwiC3b">Test Description</div>
        </div>
        '''
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        queries = ["test query"]
        result = fetch_data(queries)
        
        assert len(result) > 0
        assert result[0]["title"] == "Test Title"
        assert result[0]["description"] == "Test Description"

    @patch('requests.get', side_effect=Exception("Network error"))
    @patch('researcher_algorithm.log_error')
    def test_fetch_data_network_error(self, mock_log_error, mock_requests):
        """Test error handling in data fetching."""
        queries = ["test query"]
        result = fetch_data(queries)
        
        assert result == []
        mock_log_error.assert_called()

    @patch('researcher_algorithm.store_results')
    def test_summarize_results_with_data(self, mock_store):
        """Test result summarization with data."""
        data = [
            {"title": "Title 1", "description": "Description 1"},
            {"title": "Title 2", "description": "Description 2"}
        ]
        
        result = summarize_results(data)
        
        assert "Title 1" in result
        assert "Description 1" in result
        assert "Title 2" in result
        assert "Description 2" in result
        mock_store.assert_called_once()

    def test_summarize_results_empty_data(self):
        """Test result summarization with empty data."""
        result = summarize_results([])
        assert result == "No results found."

    @patch('researcher_algorithm.call_openai')
    def test_consult_llm_for_queries_success(self, mock_openai):
        """Test successful LLM consultation for enhanced queries."""
        task_details = "HTML5 best practices"
        base_queries = ["query1", "query2"]
        mock_openai.return_value = "enhanced query 1\nenhanced query 2\nenhanced query 3"
        
        result = consult_llm_for_queries(task_details, base_queries)
        
        assert len(result) > len(base_queries)
        assert "query1" in result
        assert "query2" in result
        mock_openai.assert_called_once()

    @patch('researcher_algorithm.call_openai', side_effect=Exception("API error"))
    @patch('researcher_algorithm.log_error')
    def test_consult_llm_for_queries_error(self, mock_log_error, mock_openai):
        """Test error handling in LLM consultation."""
        task_details = "HTML5 best practices"
        base_queries = ["query1", "query2"]
        
        result = consult_llm_for_queries(task_details, base_queries)
        
        assert result == base_queries
        mock_log_error.assert_called_once()