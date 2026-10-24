"""Tests for shorthand functionality."""

import pytest

from sktime.registry._namespace import _namespace_shorthands


def test_shorthands_unique():
    """Test that all shorthands in the namespace are unique."""
    # this call automatically raises if the shorthands do not have unique names
    _namespace_shorthands(raise_if_not_unique=True)


def test_shorthands_craftable():
    """Test that all shorthands in the namespace are safe-craftable."""
    from sktime.registry import craft

    shorthands = _namespace_shorthands()
    for _, sh in shorthands.items():
        # this would raise an exception if there is unsafe code in the shorthand
        craft(sh, safe=True)
