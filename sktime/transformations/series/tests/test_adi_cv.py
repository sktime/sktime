# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for adi_cv transformers for time series Series."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.adi_cv import ADICVTransformer


# Defining all of the categories we wish to run tests for
@pytest.mark.skipif(
    not run_test_for_class(ADICVTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "demand_series, expected_adi, expected_cv, expected_class",
    [
        ([10, 9, 10, 11, 12, 10, 9, 10, 8, 9, 10, 10], 1.09, 0.09, "smooth"),
        ([10, 20, 15, 25, 20, 30, 25, 35, 30, 40, 35, 45], 1.09, 8.90, "erratic"),
        ([6, 5, 0, 9, 2, 0, 14, 0, 0, 21, 0, 17], 1.71, 0.37, "intermittent"),
        ([0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 100], 12.0, 625.0, "lumpy"),
    ],
)
def test_adi_cv_extractor(demand_series, expected_adi, expected_cv, expected_class):
    """
    Runs a PyTest for all 4 demand time series categories.

    Parameters
    ----------
    demand_series
        A series of discrete integer values representing a demand pattern.

    expected_adi
        The expected Average Demand Interval (ADI) to be calculated by the transformer.

    expected_cv
        The expected Coefficient of Variation squared (CV^2) to be
        calculated by the transformer.

    expected_class
        The expected class to be returned by the transformer based on the
        calculated ADI and CV^2 values.
    """
    series = pd.Series(demand_series)
    transformer = ADICVTransformer()

    df = transformer.fit_transform(series)

    assert np.round(df.loc[0, "adi"], 2) == expected_adi, (
        f'The expected ADI '
        f'of {expected_adi} does not match the actual ADI of {df.loc[0, "adi"]}.'
    )

    assert np.round(df.loc[0, "cv2"], 2) == expected_cv, (
        f'The expected CV '
        f'of {expected_cv} does not match the actual CV of {df.loc[0, "cv2"]}.'
    )

    assert df.loc[0, "class"] == expected_class, (
        f'The expected class '
        f'of {expected_class} does not match the actual class of {df.loc[0, "class"]}.'
    )
