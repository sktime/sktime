"""Module containing adapters other framework packages covering multiple tasks."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["_PytsAdapter", "_TslearnAdapter"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.base.adapters._tslearn import _TslearnAdapter
