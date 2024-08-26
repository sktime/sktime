# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseDistribution", "_BaseTFDistribution"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.distributions.base import BaseDistribution
    from skpro.distributions.base._base import _BaseTFDistribution
else:
    from sktime.proba._base import BaseDistribution, _BaseTFDistribution
