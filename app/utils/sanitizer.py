"""Input sanitization utilities for the Multi-Agent Tutor.

Provides basic protection against prompt injection and
ensures user inputs are clean before reaching LLM prompts.
"""

import re


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """Sanitizes user-provided text before passing it to an LLM prompt.

    Strips common prompt injection patterns, control characters,
    and enforces a maximum length to prevent abuse.

    Args:
        text: The raw user input string.
        max_length: Maximum allowed character length. Defaults to 2000.

    Returns:
        A cleaned version of the input string.
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove common prompt injection patterns
    injection_patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)system\s*:\s*",
        r"(?i)assistant\s*:\s*",
        r"(?i)\[INST\]",
        r"(?i)\[/INST\]",
        r"(?i)<\|im_start\|>",
        r"(?i)<\|im_end\|>",
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, "", text)

    # Remove control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def sanitize_topic(topic: str) -> str:
    """Sanitizes a topic input string.

    Applies general sanitization and additionally enforces a shorter
    max length suitable for topic strings.

    Args:
        topic: The raw topic string from the user.

    Returns:
        A cleaned topic string, max 200 characters.
    """
    return sanitize_user_input(topic, max_length=200)
