#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functionality for selecting regression models."""

__all__ = [
    "TSRGridSearchCV",
]

from sktime.regression.model_selection._tune import TSRGridSearchCV