"""Centralized LLM factory for the Multi-Agent Tutor.

Provides a cached ``get_llm()`` function so all agent nodes share
a single ``ChatOpenAI`` instance instead of re-creating one per call.
"""

import os
import functools

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.utils.logger import logger

load_dotenv()


@functools.lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Creates and caches a ChatOpenAI instance from environment variables.

    Uses ``lru_cache`` so the LLM client is only instantiated once
    and reused across all agent nodes within the same process.

    Environment Variables:
        MODEL_NAME: The model identifier (e.g. ``llama3.1``).
        MODEL_BASE_URL: The API base URL (e.g. ``http://localhost:11434/v1``).
        API_KEY: The API key (e.g. ``ollama`` for local Ollama).

    Returns:
        A configured ChatOpenAI instance.

    Raises:
        ValueError: If ``API_KEY`` or ``MODEL_NAME`` are missing.
    """
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("MODEL_BASE_URL")
    api_key = os.getenv("API_KEY")

    if not api_key:
        logger.error("API_KEY not found in environment")
        raise ValueError("API_KEY is missing!")

    if not model_name:
        logger.error("MODEL_NAME not found in environment")
        raise ValueError("MODEL_NAME is missing!")

    logger.info(f"Initializing LLM: model={model_name}, base_url={base_url}")
    return ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
