"""Minimal library of kotsu methods still used in sktime.

PRIVATE MODULE: NOT FOR USE OUTSIDE OF DIRECT CALLS FROM SKTIME.

This library is a minimal, stripped down version of the kotsu library
that covers remaining imports from sktime's benchmarking module.

The module is currently undergoing a refactor with the following targets:

* remove kotsu as a soft dependency. The package is no longer maintained and
  a dependency liability for sktime, maintainers are non-responsive.
* replace kotsu with sktime's own benchmarking module, with an API that is closer
  to sklearn's estimator API (e.g., no separation of class and params).
* the refactor should cover both the time series classification benchmarks module,
  which is a native sktime implementation,
  and the newer forecasting benchmarks module, which used kotsu.
* retain proper credit to the original authors of kotsu, beyond their direct
  contributions to sktime, following open source and academic standards of crediting.

Contents are subject to MIT license of the original kotsu library:

MIT License

Copyright (c) 2021 datavaluepeople LLP.

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
