"""Python implementation of MIRA Medical Time Series Foundation Model.

Unofficial fork of the ``microsoft/MIRA`` package, maintained in ``sktime``.
This implementation is intended as a temporary solution until the authors
release a dedicated PyPI package.

sktime migration: 2026, June

Original authors: Microsoft Corporation

Code is released under the MIT License. See the MIRA repository LICENSE file.
"""

from sktime.libs.mira.configuration_mira import MIRAConfig
from sktime.libs.mira.modeling_mira import MIRAForPrediction
from sktime.libs.mira.utils_time_normalization import normalize_time_for_ctrope

__all__ = [
    "MIRAConfig",
    "MIRAForPrediction",
    "normalize_time_for_ctrope",
]
