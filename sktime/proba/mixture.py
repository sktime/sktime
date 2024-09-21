# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixture distribution."""

__author__ = ["fkiraly"]

__all__ = ["Mixture"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.mixture import Mixture
else:
    from sktime.proba._mixture import Mixture
