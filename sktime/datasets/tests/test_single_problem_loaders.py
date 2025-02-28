"""Test single problem loaders using data shipping with sktime."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import (
    load_acsf1,
    load_arrow_head,
    load_basic_motions,
    load_italy_power_demand,
    load_japanese_vowels,
    load_osuleaf,
    load_plaid,
    load_tecator,
    load_unit_test,
)

UNIVARIATE_PROBLEMS = [
    load_acsf1,
    load_arrow_head,
    load_italy_power_demand,
    load_osuleaf,
    load_unit_test,
    load_tecator,
]
MULTIVARIATE_PROBLEMS = [
    load_basic_motions,
]
UNEQUAL_LENGTH_PROBLEMS = [
    load_plaid,
    load_japanese_vowels,
]


@pytest.mark.parametrize(
    "loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS + UNEQUAL_LENGTH_PROBLEMS
)
def test_load_dataframe(loader):
    """Test that we can load all baked in TSC problems into nested pd.DataFrames."""
    # should work for all
    X, y = loader()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    X = loader(return_X_y=False)
    assert isinstance(X, pd.DataFrame)


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS)
@pytest.mark.parametrize("split", [None, "train", "test", "TRAIN", "TEST"])
def test_load_numpy3d(loader, split):
    """Test that we can load equal length TSC problems into numpy3d."""
    X, y = loader(split=split, return_type="numpy3d")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert y.ndim == 1


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS)
def test_load_numpy2d_univariate(loader):
    """Test that we can load univariate equal length TSC problems into numpy2d."""
    X, y = loader(return_type="numpy2d")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1


@pytest.mark.parametrize("loader", MULTIVARIATE_PROBLEMS)
def test_load_numpy2d_multivariate_raises(loader):
    """Test that multivariate and/or unequal length raise the correct error."""
    with pytest.raises(ValueError, match="attempting to load into a numpy2d"):
        X, y = loader(return_type="numpy2d")
