"""Shared test fixtures for the Multi-Agent Tutor test suite."""

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Sets required environment variables for all tests.

    Ensures no test accidentally hits a real LLM endpoint.
    """
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("API_KEY", "test-key")
