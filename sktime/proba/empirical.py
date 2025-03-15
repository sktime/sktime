# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution."""

__author__ = ["fkiraly"]

__all__ = ["Empirical"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.empirical import Empirical
