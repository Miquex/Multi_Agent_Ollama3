---
name: python-pytest
description: Guides the agent to write robust, maintainable unit and integration tests using the pytest framework. Use when writing, modifying, or running Python tests.
---

# Python Pytest Skill

When writing or modifying tests, you must follow these conventions and best practices for the `pytest` framework.

## When to use this skill

- Writing new unit tests or integration tests.
- Modifying existing test files.
- Setting up fixtures, mocking, or parameterization.
- Running and debugging test failures in a Python project.
- Testing asynchronous code (e.g., FastAPI endpoints, async services).

## Directory Structure and Naming Conventions

1. **Test Directory:** All tests should reside in a `tests/` directory at the root of the project, or adjacent to the module being tested.
2. **Test File Naming:** Test files MUST start with `test_` or end with `_test.py` (e.g., `test_webhook.py`).
3. **Test Function Naming:** Test function names MUST be descriptive and start with `test_` (e.g., `test_process_message_handles_valid_input`).
4. **Test Class Naming:** If grouping tests in classes, the class name MUST start with `Test` (e.g., `TestWebhookProcessor`). Do not use `__init__` in test classes.

## How to use it

### 1. Structure Tests with Arrange-Act-Assert (AAA)
Keep tests clean and readable by organizing them into three distinct sections:
- **Arrange:** Set up the test data, mocks, and fixtures.
- **Act:** Execute the code or function under test.
- **Assert:** Verify the results against expected outcomes.

```python
def test_calculate_total():
    # Arrange
    items = \[\{"price": 10\}, \{"price": 20\}\]
    
    # Act
    total = calculate_total(items)
    
    # Assert
    assert total == 30
```

### 2. Use Fixtures for Setup/Teardown
Do not use `setup()` or `teardown()` methods. Instead, use `pytest.fixture`.
- Use fixtures for dependency injection (e.g., database sessions, clients, mock data).
- Use `conftest.py` to share fixtures across multiple test files.
- Use `yield` instead of `return` in fixtures when teardown logic is required.

```python
import pytest

@pytest.fixture
def mock_db_session():
    # Setup
    session = create_session()
    yield session
    # Teardown
    session.rollback()
    session.close()
```

### 3. Asynchronous Testing (`pytest-asyncio`)
For testing `async` functions or endpoints (e.g., FastAPI):
- Mark async tests with `@pytest.mark.asyncio`, or configure `pytest-asyncio` to auto-discover async tests.
- Use `AsyncMock` from `unittest.mock` when mocking asynchronous dependencies.

```python
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_async_fetch():
    mock_client = AsyncMock()
    mock_client.get.return_value = {"status": "ok"}
    
    result = await fetch_data(mock_client)
    assert result == {"status": "ok"}
```

### 4. Parameterization
Use `@pytest.mark.parametrize` to test multiple inputs and expected outputs in a single test function, reducing code duplication.

```python
import pytest

@pytest.mark.parametrize("input_val, expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input_val, expected):
    assert square(input_val) == expected
```

### 5. Mocking and Patching
- Use `unittest.mock.patch` or the `pytest-mock` plugin (the `mocker` fixture) to isolate functionality.
- Only mock external dependencies (e.g., APIs, databases) or slow components. Try to test business logic independently whenever possible.
- Avoid over-mocking, which can lead to tests that are tightly coupled to the implementation instead of the behavior.

### 6. Exception Testing
To test that a function raises an expected exception, use `pytest.raises()`.

```python
import pytest

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        divide(10, 0)
```

## Running Tests
When instructed to run tests, use the appropriate `pytest` command, keeping verbosity and useful flags in mind:
- Basic run: `pytest`
- Verbose output: `pytest -v`
- Run a specific file: `pytest tests/test_webhook.py`
- Run a specific test function: `pytest tests/test_webhook.py::test_process_message`
- Stop on first failure: `pytest -x`
- Print output explicitly (helpful for debugging): `pytest -s`
