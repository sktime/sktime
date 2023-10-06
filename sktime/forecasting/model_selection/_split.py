#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement dataset splitting for model evaluation and selection."""

# This module is deprecated. Below the deprecation cycle :
# - Until the 0.25.0 release, imports are allowed from this location but raise a
# DeprecationWarning.
# - In the 0.25.0 release, this file (sktime/forecasting/model_selection/_split.py)
# will be removed.
# todo 0.25.0 : Please ensure this file is deleted.

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
    "Please import sktime splitters from `sktime.split`. Importing splitters from "
    "`sktime.forecasting.model_selection` is deprecated and will be removed in release "
    "0.25.0",
    DeprecationWarning,
    stacklevel=2,
)
