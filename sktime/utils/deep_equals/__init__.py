# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Module for nested equality checking."""
from sktime.utils.deep_equals._deep_equals import deep_equals
from sktime.utils.validation._dependencies import _check_soft_dependencies

__all__ = [
    "deep_equals",
]
