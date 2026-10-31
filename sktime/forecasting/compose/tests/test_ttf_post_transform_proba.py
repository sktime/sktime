#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for post-transformer application in TransformedTargetForecaster."""

__author__ = ["marrov"]
__all__ = []

import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.exponent import ExponentTransformer


def test_ttf_post_transform_applied_to_quantiles():
    """Post-transformers should apply to quantile predictions same as predict()."""

    # simple positive series
    y = pd.Series(np.arange(1, 21))

    # post-transformer squares outputs
    ttf = TransformedTargetForecaster(
        steps=[
            ("forecaster", NaiveForecaster(strategy="last")),
            ("post", ExponentTransformer(power=2)),
        ]
    )

    fh = [1, 2, 3]
    ttf.fit(y, fh=fh)

    y_pred = ttf.predict(fh)

    q_pred = ttf.predict_quantiles(fh=fh, alpha=[0.5])

    q_median = q_pred.xs(0.5, level=1, axis=1).iloc[:, 0]

    assert np.allclose(y_pred.values, q_median.values)


def test_ttf_post_transform_applied_to_interval():
    """Post-transformers should apply to quantile predictions same as predict()."""
    y = pd.Series(np.arange(1, 21))

    ttf = TransformedTargetForecaster(
        steps=[
            ("forecaster", NaiveForecaster(strategy="last")),
            ("post", ExponentTransformer(power=2)),
        ]
    )

    fh = [1, 2, 3]
    ttf.fit(y, fh=fh)

    y_pred = ttf.predict(fh)

    i_pred = ttf.predict_interval(fh=fh, coverage=[0.5])

    lower = i_pred.xs(("Coverage", 0.5, "lower"), axis=1).iloc[:, 0]

    upper = i_pred.xs(("Coverage", 0.5, "upper"), axis=1).iloc[:, 0]
    center = (lower + upper) / 2
    assert np.allclose(y_pred.values, center.values)
