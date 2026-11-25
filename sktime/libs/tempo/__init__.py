"""Python implementation of TEMPO Foundation Model for Time Series Forecasting.

Unofficial fork of the ``DC-research/TEMPO`` package,
maintained in ``sktime``.

sktime migration: 2026, September
Original authors: Defu Cao, Furong Jia, Sercan O Arik, Tomas Pfister,
Yixiang Zheng, Wen Ye, Yan Liu

MIT License

Copyright (c) 2024 Defu Cao

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

from sktime.libs.tempo.embed import DataEmbedding, DataEmbedding_wo_time
from sktime.libs.tempo.rev_in import RevIn
from sktime.libs.tempo.tempo import TEMPO

__all__ = ["TEMPO", "DataEmbedding", "DataEmbedding_wo_time", "RevIn"]
