# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseDistribution API points."""

__author__ = ["fkiraly", "Alex-JG3"]

__all__ = ["TestAllDistributions"]

from sktime.proba._error import _proba_error
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.tests.test_all_distrs import TestAllDistributions
else:
    _proba_error()
