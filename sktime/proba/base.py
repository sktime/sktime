# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseDistribution", "_BaseTFDistribution"]

from skpro.distributions.base import BaseDistribution
from skpro.distributions.base._base import _BaseTFDistribution
