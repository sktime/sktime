#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of DropNA functionality."""

__author__ = ["hliebert"]
__all__ = []

import numpy as np
import pytest

from sktime.datasets import load_longley
from sktime.transformations.series.dropna import DropNA
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal


@pytest.fixture
def X_few_na():
    """DataFrame with occasional missing."""
    _, X_few_na = load_longley()
    X_few_na.loc["1947", "GNPDEFL"] = np.nan
    X_few_na.loc["1950", "GNPDEFL"] = np.nan
    X_few_na.loc["1950", "GNP"] = np.nan
    return X_few_na


@pytest.fixture
def X_few_na_expected(X_few_na):
    """Expected results for a DataFrame with occasional missing."""
    return {
        "0": {
            "None": X_few_na.drop(labels=["1947", "1950"], axis=0),
            "any": X_few_na.drop(labels=["1947", "1950"], axis=0),
            "all": X_few_na,
        },
        "index": {
            "None": X_few_na.drop(labels=["1947", "1950"], axis=0),
            "any": X_few_na.drop(labels=["1947", "1950"], axis=0),
            "all": X_few_na,
        },
        "1": {
            "None": X_few_na.drop(labels=["GNPDEFL", "GNP"], axis=1),
            "any": X_few_na.drop(labels=["GNPDEFL", "GNP"], axis=1),
            "all": X_few_na,
        },
        "columns": {
            "None": X_few_na.drop(labels=["GNPDEFL", "GNP"], axis=1),
            "any": X_few_na.drop(labels=["GNPDEFL", "GNP"], axis=1),
            "all": X_few_na,
        },
    }


@pytest.fixture
def X_many_na():
    """DataFrame with complete columns/rows missing."""
    _, X_many_na = load_longley()
    X_many_na.loc["1947", "GNPDEFL"] = np.nan
    X_many_na.loc["1950", "GNPDEFL"] = np.nan
    X_many_na.loc["1950", "GNP"] = np.nan
    X_many_na.loc[:, "ARMED"] = np.nan
    X_many_na.loc[:, "POP"] = np.nan
    X_many_na.loc["1960", :] = np.nan
    X_many_na.loc["1962", :] = np.nan
    return X_many_na


@pytest.fixture
def X_many_na_expected(X_many_na):
    """Expected results for a DataFrame with complete columns/rows missing."""
    return {
        "0": {
            "None": X_many_na.drop(labels=X_many_na.index, axis=0),
            "any": X_many_na.drop(labels=X_many_na.index, axis=0),
            "all": X_many_na.drop(labels=["1960", "1962"]),
        },
        "index": {
            "None": X_many_na.drop(labels=X_many_na.index, axis=0),
            "any": X_many_na.drop(labels=X_many_na.index, axis=0),
            "all": X_many_na.drop(labels=["1960", "1962"]),
        },
        "1": {
            "None": X_many_na.drop(labels=X_many_na.columns, axis=1),
            "any": X_many_na.drop(labels=X_many_na.columns, axis=1),
            "all": X_many_na.drop(labels=["ARMED", "POP"], axis=1),
        },
        "columns": {
            "None": X_many_na.drop(labels=X_many_na.columns, axis=1),
            "any": X_many_na.drop(labels=X_many_na.columns, axis=1),
            "all": X_many_na.drop(labels=["ARMED", "POP"], axis=1),
        },
    }


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", DropNA.VALID_HOW_VALUES)
def test_dropna_few_na(axis, how, X_few_na, X_few_na_expected):
    """Test expected results on a DataFrame with occasional missing."""
    transformer = DropNA(axis=axis, how=how, thresh=None)
    X_transformed = transformer.fit_transform(X_few_na)
    X_expected = X_few_na_expected[str(axis)][str(how)]

    _assert_array_almost_equal(X_transformed, X_expected)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", DropNA.VALID_HOW_VALUES)
def test_dropna_many_na(axis, how, X_many_na, X_many_na_expected):
    """Test expected results on a DataFrame with complete columns/rows missing."""
    transformer = DropNA(axis=axis, how=how, thresh=None)
    X_transformed = transformer.fit_transform(X_many_na)
    X_expected = X_many_na_expected[str(axis)][str(how)]

    _assert_array_almost_equal(X_transformed, X_expected)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", ["any", "all"])
@pytest.mark.parametrize("thresh", [10, 0.5])
@pytest.mark.parametrize("remember", [None, True, False])
def test_dropna_conflicting_arguments(axis, how, thresh, remember):
    """Test that how and thresh arguments cannot be both set."""
    with pytest.raises(TypeError, match=r"thresh cannot be set together with how"):
        DropNA(axis=axis, how=str(how), thresh=thresh, remember=remember)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", [True, False, "invalid_value", [1, 2, 3]])
@pytest.mark.parametrize("remember", [None, True, False])
def test_dropna_invalid_arguments_how(axis, how, remember):
    """Test that invalid arguments for how are not accepted."""
    with pytest.raises(ValueError):
        DropNA(axis=axis, how=how, thresh=None, remember=remember)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("thresh", [True, False, "invalid_value", [1, 2, 3]])
@pytest.mark.parametrize("remember", [None, True, False])
def test_dropna_invalid_arguments_thresh_type(axis, thresh, remember):
    """Test that invalid arguments for thresh are not accepted."""
    with pytest.raises(TypeError):
        DropNA(axis=axis, how=None, thresh=thresh, remember=remember)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("thresh", [-5, -3.5, 1.5])
@pytest.mark.parametrize("remember", [None, True, False])
def test_dropna_invalid_arguments_thresh_value(axis, thresh, remember):
    """Test that arguments outside sensible range for thresh are not accepted."""
    with pytest.raises(ValueError):
        DropNA(axis=axis, how=None, thresh=thresh)
