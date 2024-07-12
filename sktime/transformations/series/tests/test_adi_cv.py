# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for adi_cv transformers for time series Series."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.adi_cv import ADICVTransformer


def _generate_smooth_series(size: int = 750):
    """Generates a demand time series of the "smooth" category.

    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating a smooth series, we keep variance low by keeping the
    # standard deviation to 0.25. Denoted as scale according to numpy docs

    smooth_series = np.random.normal(loc=10, scale=0.25, size=size)

    return pd.Series(smooth_series)


def _generate_erratic_series(size: int = 750):
    """Generates a demand time series of the "erratic" category.

    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating an erratic series, we keep variance high by keeping the
    # standard deviation to 1 and then squaring the values

    erratic_series = np.random.normal(loc=10, scale=2.5, size=size) ** 2

    return pd.Series(erratic_series)


def _generate_intermittent_series(size: int = 750):
    """Generates a demand time series of the "intermittent" category.

    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating an intermittent series, we keep ADI high by
    # setting only 10% of all values to non-zero values

    intermittent_series = np.zeros(shape=(size,))
    non_zero_indices = np.random.choice(size, size=size // 10, replace=False)

    intermittent_series[non_zero_indices] = np.random.normal(10, 0.25, size=size // 10)

    return pd.Series(intermittent_series)


def _generate_lumpy_series(size: int = 750):
    """Generates a demand time series of the "lumpy" category.

    Parameters
    ----------
    size : int, optional
        The size of the generated time series, by default 750

    Returns
    -------
    Pandas.Series
        Returns the generated series in the Pandas Series format.
    """
    # Generating a lumpy series, we keep ADI high by
    # setting only 10% of all values to non-zero values

    lumpy_series = np.zeros(shape=(size,))
    non_zero_indices = np.random.choice(size, size=size // 10, replace=False)

    lumpy_series[non_zero_indices] = np.random.normal(10, 2.5, size=size // 10) ** 2

    return pd.Series(lumpy_series)


# Defining all of the categories we wish to run tests for
@pytest.mark.skipif(
    not run_test_for_class(ADICVTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "series_generator, expected_class",
    [
        (_generate_smooth_series, "smooth"),
        (_generate_erratic_series, "erratic"),
        (_generate_intermittent_series, "intermittent"),
        (_generate_lumpy_series, "lumpy"),
    ],
)
def test_adi_cv_extractor(series_generator, expected_class):
    """
    Runs a PyTest for all 4 demand time series categories.

    Parameters
    ----------
    series_generator
        A function that generates a time series in the Pandas Series format.

    expected_class
        The expected class to be predicted by the transformer.
    """
    series = series_generator()
    transformer = ADICVTransformer()

    df = transformer.fit_transform(series)
    assert df["class"].iloc[0] == expected_class
