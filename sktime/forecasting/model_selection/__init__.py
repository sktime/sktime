#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functionality for selecting forecasting models."""

__all__ = [
    "ForecastingGridSearchCV",
    "ForecastingRandomizedSearchCV",
    "ForecastingSkoptSearchCV",
    "ForecastingOptunaSearchCV",
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "temporal_train_test_split",
]

from sktime.forecasting.model_selection._tune import (
    ForecastingGridSearchCV,
    ForecastingOptunaSearchCV,
    ForecastingRandomizedSearchCV,
    ForecastingSkoptSearchCV,
)


# todo 0.32.0 - check whether we should remove, otherwise bump
# still used in blog posts and old tutorials
def temporal_train_test_split(
    y, X=None, test_size=None, train_size=None, fh=None, anchor="start"
):
    """Split time series data into temporal train and test sets.

    DEPRECATED - use sktime.split.temporal_train_test_split instead.
    """
    from warnings import warn

    from sktime.split import temporal_train_test_split as _tts

    warn(
        "WARNING - the old location of temporal_train_test_split in "
        "sktime.forecasting.model_selection is deprecated and is scheduled for "
        "imminent removal in a MINOR version. "
        "Please update any import statements to "
        "from sktime.split import temporal_train_test_split.",
        DeprecationWarning,
    )

    return _tts(
        y=y, X=X, test_size=test_size, train_size=train_size, fh=fh, anchor=anchor
    )


# todo 0.32.0 - check whether we should remove, otherwise bump
# still used in blog posts and old tutorials
def ExpandingWindowSplitter(fh=1, initial_window=10, step_length=1):
    """Legacy export of Expanding window splitter.

    DEPRECATED - use sktime.split.ExpandingWindowSplitter instead.
    """
    from warnings import warn

    from sktime.split import ExpandingWindowSplitter as _EWSplitter

    warn(
        "WARNING - the old location of ExpandingWindowSplitter in "
        "sktime.forecasting.model_selection is deprecated and is scheduled for "
        "imminent removal in a MINOR version. "
        "Please update any import statements to "
        "from sktime.split import ExpandingWindowSplitter.",
        DeprecationWarning,
    )

    return _EWSplitter(fh=fh, initial_window=initial_window, step_length=step_length)


# todo 0.32.0 - check whether we should remove, otherwise bump
# still used in blog posts and old tutorials
def SlidingWindowSplitter(
    fh=1, window_length=10, step_length=1, initial_window=None, start_with_window=True
):
    """Legacy export of Sliding window splitter.

    DEPRECATED - use sktime.split.ExpandingWindowSplitter instead.
    """
    from warnings import warn

    from sktime.split import SlidingWindowSplitter as _SWSplitter

    warn(
        "WARNING - the old location of SlidingWindowSplitter in "
        "sktime.forecasting.model_selection is deprecated and is scheduled for "
        "imminent removal in a MINOR version. "
        "Please update any import statements to "
        "from sktime.split import SlidingWindowSplitter.",
        DeprecationWarning,
    )

    return _SWSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        initial_window=initial_window,
        start_with_window=start_with_window,
    )
