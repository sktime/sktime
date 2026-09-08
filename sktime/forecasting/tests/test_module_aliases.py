# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the deprecated module aliases in sktime.forecasting."""

import subprocess
import sys
from importlib import import_module

import pytest

from sktime.forecasting import _MODULE_ALIASES


def test_aliases_are_not_imported_eagerly():
    """Test that importing sktime.forecasting does not import the aliases.

    See issue #11027. The aliases used to be registered by importing every
    renamed module at ``sktime.forecasting`` import time, which pulled in the
    deep learning modules whether or not they were asked for.
    """
    code = (
        "import sys, sktime.forecasting\n"
        "from sktime.forecasting import _MODULE_ALIASES\n"
        "loaded = [n for n in _MODULE_ALIASES.values()"
        " if f'sktime.forecasting.{n}' in sys.modules]\n"
        "print(loaded)\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", (
        f"importing sktime.forecasting eagerly imported {result.stdout.strip()}"
    )


@pytest.mark.parametrize("old_name", sorted(_MODULE_ALIASES))
def test_alias_submodule_import(old_name):
    """Test that ``import sktime.forecasting.<old_name>`` still works."""
    new_name = _MODULE_ALIASES[old_name]

    with pytest.warns(FutureWarning, match="deprecated and has been renamed"):
        aliased = import_module(f"sktime.forecasting.{old_name}")

    renamed = import_module(f"sktime.forecasting.{new_name}")

    # the alias must be the very same module object, not a second copy
    assert aliased is renamed


@pytest.mark.parametrize("old_name", sorted(_MODULE_ALIASES))
def test_alias_attribute_access(old_name):
    """Test that ``from sktime.forecasting import <old_name>`` still works."""
    import sktime.forecasting

    with pytest.warns(FutureWarning, match="deprecated and has been renamed"):
        aliased = getattr(sktime.forecasting, old_name)

    renamed = import_module(f"sktime.forecasting.{_MODULE_ALIASES[old_name]}")

    assert aliased is renamed


def test_unknown_submodule_still_raises():
    """Test that the alias finder does not swallow genuinely missing modules."""
    with pytest.raises(ModuleNotFoundError):
        import_module("sktime.forecasting.this_module_does_not_exist")
