# -*- coding: utf-8 -*-
"""Test single problem loaders with varying return types."""
import numpy as np
import pandas as pd
import pytest

from sktime._contrib.datasets import (  # Univariate; Unequal length; Multivariate
    load_acsf1,
    load_arrow_head,
    load_basic_motions,
    load_italy_power_demand,
    load_japanese_vowels,
    load_osuleaf,
    load_plaid,
    load_unit_test,
)

UNIVARIATE_PROBLEMS = [
    load_acsf1,
    load_arrow_head,
    load_italy_power_demand,
    load_osuleaf,
    load_unit_test,
]
MULTIVARIATE_PROBLEMS = [
    load_basic_motions,
]
UNEQUAL_LENGTH_PROBLEMS = [
    load_plaid,
    load_japanese_vowels,
]


def test_load_dataframe():
    """Test that we can load all baked in TSC problems into nested pd.DataFrames."""
    # should work for all
    for loader in UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS + UNEQUAL_LENGTH_PROBLEMS:
        X, y = loader()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
    for loader in UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS + UNEQUAL_LENGTH_PROBLEMS:
        X = loader(return_X_y=False)
        assert isinstance(X, pd.DataFrame)


def test_load_numpy3d():
    """Test that we can load equal length TSC problems into numpy3d."""
    for loader in UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS:
        for spl in [None, "train", "test"]:
            X, y = loader(split=spl, return_type="numpy3d")
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.ndim == 3
            assert y.ndim == 1


# should work for all


def test_load_numpy2d():
    """Test that we can load univariate equal length TSC problems into numpy2d.

    Also test that multivariate and/or unequal length raise the correct error.
    """
    for loader in UNIVARIATE_PROBLEMS:
        X, y = loader(return_type="numpy2d")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1

    for loader in MULTIVARIATE_PROBLEMS:
        with pytest.raises(ValueError, match="attempting to load into a numpy2d"):
            X, y = loader(return_type="numpy2d")
