---
name: python-pep484-typing
description: Guides the agent to implement robust, PEP 484-compliant type hinting in Python. Use when writing new code, refactoring, or improving type safety.
---

# Python PEP 484 Typing Skill

When writing or refactoring Python code, use this skill to ensure consistent and accurate type hinting according to PEP 484 standards.

## Basic Principles

1. **Variables**: Use type hints for variable declarations when the type is not immediately obvious from the assignment.
2. **Function Signatures**: Always type hint function parameters and return values.
3. **Collections**: Use `List`, `Dict`, `Set`, and `Tuple` from the `typing` module for older Python versions, or built-ins for Python 3.9+.
4. **Complex Types**: Use `Union`, `Optional`, `Any`, `Callable`, and `Iterable` where appropriate.

## Type Hinting Rules

### Primitive Types
```python
name: str = "Alice"
age: int = 30
is_active: bool = True
price: float = 19.99
```

### Collections (Python 3.9+)
```python
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 100, "Bob": 90}
coordinates: tuple[float, float] = (40.7128, -74.0060)
tags: set[str] = {"python", "coding"}
```

### Optional and Union
Use `Optional[T]` for values that can be `T` or `None`. Use `Union[T1, T2]` for multiple possible types (or `T1 | T2` in 3.10+).

```python
from typing import Optional, Union

def process_data(data: Optional[str]) -> Union[int, float]:
    if data is None:
        return 0.0
    return len(data)
```

### Classes and Self
For methods returning an instance of the class itself, use `"ClassName"` (as a string) or `from __future__ import annotations`.

```python
class Node:
    def __init__(self, value: int):
        self.value = value
        self.next: Optional["Node"] = None
```

## How to Apply Typing

- **Analyze Flow**: Determine the expected types of inputs and outputs by checking usages.
- **Library Signatures**: When using third-party libraries, refer to their type stubs or documentation.
- **Avoid Over-typing**: Don't use `Any` unless absolutely necessary. Be as specific as possible.
- **Consistency**: Ensure that type hints in the docstrings (if included) match the actual PEP 484 annotations.

## Reviewing for Type Safety

- Does every function have parameter and return type hints?
- Are collection types narrowed (e.g., `list[str]` instead of just `list`)?
- Is `Optional` used correctly for values that could be `None`?
- Are there any inconsistent types that might cause static analysis errors?
