# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for adi_cv transformers for time series Series."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.adi_cv import ADICVTransformer


def _generate_smooth_series(size: int = 750, seed: int = 42):
    """Generates a demand time series of the "smooth" category.
    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    seed : int, optional
        The random seed state for the randomization algorithm.

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating a smooth series, we keep variance low by keeping the
    # standard deviation to 0.25. Denoted as scale according to numpy docs

    np.random.seed(seed)
    smooth_series = np.random.normal(loc=10, scale=0.25, size=12)

    return pd.Series(smooth_series)


def _generate_erratic_series(size: int = 750, seed: int = 42):
    """Generates a demand time series of the "erratic" category.
    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    seed : int, optional
        The random seed state for the randomization algorithm.

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating an erratic series, we keep variance high by keeping the
    # standard deviation to 1 and then squaring the values

    np.random.seed(seed)
    erratic_series = np.random.uniform(low=1, high=100, size=size) ** 2

    return pd.Series(erratic_series)


def _generate_intermittent_series(size: int = 750, seed: int = 42):
    """Generates a demand time series of the "intermittent" category.
    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    seed : int, optional
        The random seed state for the randomization algorithm.

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating an intermittent series, we keep ADI high by
    # setting only 10% of all values to non-zero values

    np.random.seed(seed)
    intermittent_series = np.zeros(shape=(size,))
    non_zero_indices = np.random.choice(size, size=size // 10, replace=False)

    intermittent_series[non_zero_indices] = np.random.normal(10, 0.25, size=size // 10)

    return pd.Series(intermittent_series)


def _generate_lumpy_series(size: int = 750, seed: int = 42):
    """Generates a demand time series of the "lumpy" category.
    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    seed : int, optional
        The random seed state for the randomization algorithm.

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating a lumpy series, we keep ADI high by
    # setting only 10% of all values to non-zero values

    np.random.seed(seed)
    lumpy_series = np.zeros(shape=(size,))
    non_zero_indices = np.random.choice(size, size=size // 10, replace=False)

    lumpy_series[non_zero_indices] = np.random.uniform(1, 100, size=size // 10) ** 2

    return pd.Series(lumpy_series)


# Defining all of the categories we wish to run tests for
@pytest.mark.skipif(
    not run_test_for_class(ADICVTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "demand_series, expected_adi, expected_cv, expected_class, adi_trim_handling",
    [
        (_generate_smooth_series(), 1.0, 0.0, "smooth", "pool"),
        (_generate_erratic_series(), 1.0, 0.82, "erratic", "pool"),
        (_generate_intermittent_series(), 10.0, 0.0, "intermittent", "pool"),
        (_generate_lumpy_series(), 10.0, 0.64, "lumpy", "pool"),
        (_generate_intermittent_series(), 9.95, 0.0, "intermittent", "trim"),
        (_generate_erratic_series(), 1.0, 0.82, "erratic", "ignore"),
        ([10, 9, 10, 11, 12, 10, 9, 10, 8, 9, 10, 10], 1.0, 0.01, "smooth", "pool"),
        ([10, 20, 5, 25, 20, 5, 50, 35, 30, 100, 35, 45], 1.0, 0.62, "erratic", "pool"),
        ([6, 5, 0, 9, 2, 0, 14, 0, 0, 21, 0, 17], 1.71, 0.37, "intermittent", "pool"),
        ([1, 0, 0, 50, 0, 0, 200, 0, 0, 0, 0, 100], 3.0, 0.70, "lumpy", "pool"),
    ],
)
def test_adi_cv_extractor(
    demand_series, expected_adi, expected_cv, expected_class, adi_trim_handling
):
    """
    Runs a PyTest for all 4 demand time series categories.

    Parameters
    ----------
    demand_series
        A series of discrete integer values representing a demand pattern. The series
        can be either static or randomly generated.

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
    transformer = ADICVTransformer(adi_trim_handling=adi_trim_handling)

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
