# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Laplace probability distribution."""

__author__ = ["fkiraly"]

__all__ = ["Laplace"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.laplace import Laplace
else:
    from sktime.proba._error import _proba_error as Laplace
