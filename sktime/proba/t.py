# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Student's t-distribution."""

__author__ = ["Alex-JG3", "ivarzap"]

__all__ = ["TDistribution"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.t import TDistribution
else:
    from sktime.proba._error import _proba_error as TDistribution
