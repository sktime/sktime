"""Python implementations of Kalman Filters and Kalman Smoothers.

Unofficial fork of the ``pykalman`` package, maintained in ``sktime``.

sktime migration: 2025, April
Version 0.51.0 release: 2022, Jan 22 (CyberZHG)

Original authors: CyberZHG

2024 release subject to following license:

Copyright (c) 2018 PoW

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
SOFTWARE.
"""

__authors__ = ["CyberZHG", "EtcEc"]

from sktime.libs._keras_self_attention.real_former import (
    ResidualScaledDotProductAttention,
)
from sktime.libs._keras_self_attention.scaled_dot_attention import (
    ScaledDotProductAttention,
)
from sktime.libs._keras_self_attention.seq_self_attention import SeqSelfAttention
from sktime.libs._keras_self_attention.seq_weighted_attention import (
    SeqWeightedAttention,
)

__all__ = [
    "SeqSelfAttention",
    "SeqWeightedAttention",
    "ScaledDotProductAttention",
    "ResidualScaledDotProductAttention",
]
