#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functionality for selecting forecasting models."""

__all__ = [
    "ForecastingGridSearchCV",
    "ForecastingRandomizedSearchCV",
    "ForecastingSkoptSearchCV",
    "temporal_train_test_split",
]

from sktime.forecasting.model_selection._tune import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    ForecastingSkoptSearchCV,
)


# todo 0.26.0 - check whether we should remove, otherwise bump
def temporal_train_test_split(
    y, X=None, test_size=None, train_size=None, fh=None, anchor="start"
):
    """Split time series data into temporal train and test sets."""
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
