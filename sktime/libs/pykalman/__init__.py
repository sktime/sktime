"""Python implementations of Kalman Filters and Kalman Smoothers.

Unofficial fork of the ``pykalman`` package, maintained in ``sktime``.

sktime migration: 2024, June
Version 0.9.7 release: 2024, Mar 25 (mbalatsko, fkiraly)
Version 0.9.5 release: 2013, Jul 7 (duckworthd)

Original authors: Daniel Duckworth

2013 and 2024 releases subject to following license:

All code contained except that in pykalman/utils.py is released under the
license below. All code in pykalman/utils.py is released under the license
contained therein.

New BSD License

Copyright (c) 2012 Daniel Duckworth.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Daniel Duckworth nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

__authors__ = [
    "duckworthd",  # main author original pykalman package
    "mbalatsko",  # update ot python 3.11 and later, temporary maintainer
    "gliptak",  # minor updates
    "nils-werner",  # minor updates
    "jonathanng",  # minor docs fix
    "pierre-haessig",  # minor docs fix
]

from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

__all__ = [
    "KalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt",
]
