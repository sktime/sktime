"""BATS and TBATS time series forecasting.

Unofficial fork of the ``tbats`` package, maintained in ``sktime``.

sktime migration: 2026, September (cotterpl)
Version 1.1.3 release: 2023, Apr 17 (cotterpl)

Original authors: Grzegorz Skorupa

The 2023 release subject to following license:

MIT License

Copyright (C) 2019 intive (intive.com)

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
from . import abstract, bats, tbats
from .bats import BATS
from .tbats import TBATS

__all__ = [
    "BATS",
    "TBATS",
    "bats",
    "tbats",
    "abstract",
]
