"""Tests for the LLM factory module."""

from unittest.mock import patch, MagicMock

import pytest

from app.core.llm_factory import get_llm


class TestGetLlm:
    """Tests for the get_llm factory function."""

    def setup_method(self):
        """Clear the LRU cache before each test to ensure isolation."""
        get_llm.cache_clear()

    def test_missing_api_key_raises(self, monkeypatch):
        """Should raise ValueError when API_KEY is not set."""
        monkeypatch.delenv("API_KEY", raising=False)

        with pytest.raises(ValueError, match="API_KEY is missing"):
            get_llm()

    def test_missing_model_name_raises(self, monkeypatch):
        """Should raise ValueError when MODEL_NAME is not set."""
        monkeypatch.delenv("MODEL_NAME", raising=False)

        with pytest.raises(ValueError, match="MODEL_NAME is missing"):
            get_llm()

    @patch("app.core.llm_factory.ChatOpenAI")
    def test_creates_instance_with_env_vars(self, mock_chat, monkeypatch):
        """Should create ChatOpenAI with correct parameters from env."""
        # Arrange
        monkeypatch.setenv("MODEL_NAME", "llama3.1")
        monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("API_KEY", "test-key")

        # Act
        get_llm()

        # Assert
        mock_chat.assert_called_once_with(
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
        )

    @patch("app.core.llm_factory.ChatOpenAI")
    def test_caching_returns_same_instance(self, mock_chat):
        """Subsequent calls should return the cached instance, not re-create."""
        # Arrange
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        # Act
        result1 = get_llm()
        result2 = get_llm()

        # Assert — ChatOpenAI should only be called once due to caching
        assert mock_chat.call_count == 1
        assert result1 is result2
