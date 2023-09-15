#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement dataset splitting for model evaluation and selection."""

__all__ = [
    "ExpandingGreedySplitter",
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "CutoffSplitter",
    "SingleWindowSplitter",
    "SameLocSplitter",
    "temporal_train_test_split",
    "TestPlusTrainSplitter",
]

import warnings

from sktime.split import (
    CutoffSplitter,
    ExpandingGreedySplitter,
    ExpandingWindowSplitter,
    SameLocSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    TestPlusTrainSplitter,
    temporal_train_test_split,
)

warnings.warn(
    "The 'sktime.forecasting.model_selection' module is deprecated for splitter "
    "functions. Please use 'sktime.split' instead.",
    DeprecationWarning,
    stacklevel=2,
)
