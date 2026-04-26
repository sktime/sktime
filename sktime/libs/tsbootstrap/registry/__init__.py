"""Registry and lookup functionality."""

from tsbootstrap.registry._lookup import all_objects
from tsbootstrap.registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
)

__all__ = [
    "OBJECT_TAG_LIST",
    "OBJECT_TAG_REGISTER",
    "all_objects",
]
