"""Python implementation of MIRA Medical Time Series Foundation Model.

Unofficial fork of the ``microsoft/MIRA`` package, maintained in ``sktime``.
This implementation is intended as a temporary solution until the authors
release a dedicated PyPI package.

sktime migration: 2026, June

Original authors: Microsoft Corporation

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""

from sktime.libs.mira.configuration_mira import MIRAConfig
from sktime.libs.mira.mira_inference import mira_predict_autoregressive_norm
from sktime.libs.mira.modeling_mira import MIRAForPrediction
from sktime.libs.mira.utils_time_normalization import normalize_time_for_ctrope

__all__ = [
    "MIRAConfig",
    "MIRAForPrediction",
    "mira_predict_autoregressive_norm",
    "normalize_time_for_ctrope",
]
