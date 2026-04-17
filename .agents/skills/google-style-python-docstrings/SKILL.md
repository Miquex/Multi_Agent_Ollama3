---
name: google-style-python-docstrings
description: Guides the agent to write consistent, clear, and comprehensive Python docstrings adhering strictly to the Google Python Style Guide. Use when writing, refactoring, or reviewing Python documentation.
---

# Google Style Python Docstrings Skill

When writing or reviewing Python code, follow these steps to ensure docstrings comply with the Google Python Style Guide.

## Docstring Formatting Rules

1. **Summary Line**: The first line should be a brief, one-line summary of what the function, class, or module does. The summary should end with a period.
2. **Blank Lines**: The summary line should be followed by a blank line before any detailed description or sections.
3. **Sections**: Use appropriate sections labeled with a specific header followed by a colon (e.g., `Args:`, `Returns:`, `Raises:`).
4. **Indentation**: Descriptions under sections (like `Args:`) must be indented by 2 or 4 spaces consistently.

## Key Sections

- **Args:** List each parameter by name. Follow the name with its type in parentheses (if not type-hinted in the signature), a colon, and its description.
- **Returns:** (or **Yields:** for generators) Describe the return value and its type. If there are multiple return values, describe them as a tuple.
- **Raises:** List all exceptions that are explicitly raised and provide the conditions under which they occur.
- **Attributes:** For classes, list public attributes exposed by the class.

## Examples

### Functions & Methods
```python
def fetch_user_data(user_id: int, include_history: bool = False) -> dict:
    """Fetches user data from the database.

    Retrieves the user's profile and optionally their activity history.

    Args:
        user_id: The unique identifier of the user. (Note: no type here if signature is typed).
        include_history: Whether to include the user's activity history.
            Defaults to False.

    Returns:
        A dictionary containing user profile information.

    Raises:
        ValueError: If the user_id is negative.
        ConnectionError: If the database connection fails.
    """
    pass
```

### Classes
```python
class DatabaseConnection:
    """Manages connections to the database.

    This class handles connecting, disconnecting, and querying the database ensuring
    safe resource management.

    Attributes:
        connection_url (str): The URL used to connect to the database.
        is_connected (bool): Indicates if the connection is currently active.
    """
    pass
```

## How to Review Code for Docstrings

- **Completeness**: Are there docstrings for all public modules, functions, classes, and methods?
- **Accuracy**: Do the documented arguments and return types match the actual function signature?
- **Formatting**: Does the docstring use the correct Google Style headers and indentation?
- **Clarity**: Is the description easy to understand and free of obvious, redundant statements?
