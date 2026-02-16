# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for dependency checking utilities."""

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_for_class(_check_soft_dependencies),
    reason="run test incrementally (if requested)",
)
def test_check_soft_dependencies():
    """Test check_soft_dependencies."""
    ALWAYS_INSTALLED = "sktime"
    ALWAYS_INSTALLED2 = "numpy"
    ALWAYS_INSTALLED_W_V = "sktime>=0.5.0"
    ALWAYS_INSTALLED_W_V2 = "numpy>=0.1.0"
    NEVER_INSTALLED = "nonexistent__package_foo_bar"
    NEVER_INSTALLED_W_V = "sktime<0.1.0"

    # Test that the function does not raise an error when all dependencies are installed
    _check_soft_dependencies(ALWAYS_INSTALLED)
    _check_soft_dependencies(ALWAYS_INSTALLED, ALWAYS_INSTALLED2)
    _check_soft_dependencies(ALWAYS_INSTALLED_W_V)
    _check_soft_dependencies(ALWAYS_INSTALLED_W_V, ALWAYS_INSTALLED_W_V2)
    _check_soft_dependencies(ALWAYS_INSTALLED, ALWAYS_INSTALLED2, ALWAYS_INSTALLED_W_V2)
    _check_soft_dependencies([ALWAYS_INSTALLED, ALWAYS_INSTALLED2])

    # Test that error is raised when a dependency is not installed
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(NEVER_INSTALLED)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(NEVER_INSTALLED, ALWAYS_INSTALLED)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([ALWAYS_INSTALLED, NEVER_INSTALLED])
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(ALWAYS_INSTALLED, NEVER_INSTALLED_W_V)
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([ALWAYS_INSTALLED, NEVER_INSTALLED_W_V])

    # disjunction cases, "or" - positive cases
    _check_soft_dependencies([[ALWAYS_INSTALLED, NEVER_INSTALLED]])
    _check_soft_dependencies(
        [
            [ALWAYS_INSTALLED, NEVER_INSTALLED],
            [ALWAYS_INSTALLED_W_V, NEVER_INSTALLED_W_V],
            ALWAYS_INSTALLED2,
        ]
    )

    # disjunction cases, "or" - negative cases
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies([[NEVER_INSTALLED, NEVER_INSTALLED_W_V]])
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                [NEVER_INSTALLED, NEVER_INSTALLED_W_V],
                [ALWAYS_INSTALLED, NEVER_INSTALLED],
                ALWAYS_INSTALLED2,
            ]
        )
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                ALWAYS_INSTALLED2,
                [ALWAYS_INSTALLED, NEVER_INSTALLED],
                NEVER_INSTALLED_W_V,
            ]
        )
    with pytest.raises(ModuleNotFoundError):
        _check_soft_dependencies(
            [
                [ALWAYS_INSTALLED, ALWAYS_INSTALLED2],
                NEVER_INSTALLED,
                ALWAYS_INSTALLED2,
            ]
        )
