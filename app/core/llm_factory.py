"""Centralized LLM factory for the Multi-Agent Tutor.

Provides a cached ``get_llm()`` function and a pre-flight ``check_ollama_health()``
to ensure the local inference environment is ready.
"""

import os
import json
import functools
import urllib.request
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.utils.logger import logger

load_dotenv()


def check_ollama_health() -> bool:
    """Verifies that the Ollama service is reachable and the model is downloaded.

    Extracts the host from ``MODEL_BASE_URL`` and queries the Ollama ``/api/tags``
    endpoint. This is a synchronization guard to prevent runtime errors
    during agent execution.

    Returns:
        True if the health check passes or if using a non-local provider.
        False if Ollama is unreachable or the model is missing.
    """
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("MODEL_BASE_URL", "")
    api_key = os.getenv("API_KEY", "")

    # Skip health check if likely using a cloud provider like OpenAI
    if "localhost" not in base_url and "127.0.0.1" not in base_url and api_key != "ollama":
        logger.info("Non-local LLM detected; skipping Ollama health check.")
        return True

    if not model_name:
        logger.error("MODEL_NAME is not set in environment.")
        return False

    # Derive the native Ollama API endpoint from the v1 base URL
    # e.g. http://localhost:11434/v1 -> http://localhost:11434/api/tags
    parsed_url = urlparse(base_url)
    api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/tags"

    logger.info(f"Performing pre-flight health check on Ollama at {api_base}...")

    try:
        with urllib.request.urlopen(api_base, timeout=5) as response:
            if response.getcode() != 200:
                logger.error(f"Ollama health check failed: HTTP {response.getcode()}")
                return False
            data = json.loads(response.read().decode())
    except Exception as e:
        logger.error(
            f"Ollama service unreachable at {api_base}. "
            f"Ensure the Ollama service is running. Error: {e}"
        )
        return False

    # Check if the requested model (or its :latest variant) is in the tags list
    local_models = [m.get("name") for m in data.get("models", [])]
    if model_name in local_models or f"{model_name}:latest" in local_models:
        logger.success(f"Health check passed: Model '{model_name}' is ready.")
        return True

    logger.error(
        f"Model '{model_name}' not found in your local Ollama library. "
        f"Available: {local_models}"
    )
    logger.info(f"Fix this by running: ollama pull {model_name}")
    return False


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
    return ChatOpenAI(model=model_name, base_url=base_url, api_key=SecretStr(api_key))
