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

# todo: univariate case (implement check that axis=0 first).

y_few_na, X_few_na = load_longley()
X_few_na.loc["1947", "GNPDEFL"] = np.nan
X_few_na.loc["1950", "GNPDEFL"] = np.nan
X_few_na.loc["1950", "GNP"] = np.nan

X_few_na_expected = {
    "0": {
        "None": X_few_na.drop(
            labels=[
                "1947",
                "1950",
            ],
            axis=0,
        ),
        "any": X_few_na.drop(
            labels=[
                "1947",
                "1950",
            ],
            axis=0,
        ),
        "all": X_few_na,
    },
    "index": {
        "None": X_few_na.drop(
            labels=[
                "1947",
                "1950",
            ],
            axis=0,
        ),
        "any": X_few_na.drop(
            labels=[
                "1947",
                "1950",
            ],
            axis=0,
        ),
        "all": X_few_na,
    },
    "1": {
        "None": X_few_na.drop(
            labels=[
                "GNPDEFL",
                "GNP",
            ],
            axis=1,
        ),
        "any": X_few_na.drop(
            labels=[
                "GNPDEFL",
                "GNP",
            ],
            axis=1,
        ),
        "all": X_few_na,
    },
    "columns": {
        "None": X_few_na.drop(
            labels=[
                "GNPDEFL",
                "GNP",
            ],
            axis=1,
        ),
        "any": X_few_na.drop(
            labels=[
                "GNPDEFL",
                "GNP",
            ],
            axis=1,
        ),
        "all": X_few_na,
    },
}

y_many_na, X_many_na = load_longley()
X_many_na.loc["1947", "GNPDEFL"] = np.nan
X_many_na.loc["1950", "GNPDEFL"] = np.nan
X_many_na.loc["1950", "GNP"] = np.nan
X_many_na.loc[:, "ARMED"] = np.nan
X_many_na.loc[:, "POP"] = np.nan
X_many_na.loc["1960", :] = np.nan
X_many_na.loc["1962", :] = np.nan
X_many_na.dropna(axis=1)

X_many_na_expected = {
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
        "all": X_many_na.drop(
            labels=[
                "ARMED",
                "POP",
            ],
            axis=1,
        ),
    },
    "columns": {
        "None": X_many_na.drop(labels=X_many_na.columns, axis=1),
        "any": X_many_na.drop(labels=X_many_na.columns, axis=1),
        "all": X_many_na.drop(
            labels=[
                "ARMED",
                "POP",
            ],
            axis=1,
        ),
    },
}


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", DropNA.VALID_HOW_VALUES)
def test_dropna_few_na(axis, how):
    """Test expected results on a DataFrame with occasional missings."""
    transformer = DropNA(axis=axis, how=how, thresh=None)
    X_transformed = transformer.fit_transform(X_few_na)
    X_expected = X_few_na_expected[str(axis)][str(how)]

    _assert_array_almost_equal(X_transformed, X_expected)


@pytest.mark.parametrize("axis", DropNA.VALID_AXIS_VALUES)
@pytest.mark.parametrize("how", DropNA.VALID_HOW_VALUES)
def test_dropna_many_na(axis, how):
    """Test expected results on a DataFrame with complete columns/rows missing."""
    transformer = DropNA(axis=axis, how=how, thresh=None)
    X_transformed = transformer.fit_transform(X_many_na)
    X_expected = X_many_na_expected[str(axis)][str(how)]

    _assert_array_almost_equal(X_transformed, X_expected)


@pytest.mark.parametrize("how", ["any", "all"])
@pytest.mark.parametrize("thresh", [10, 0.5])
def test_dropna_incompatible_arguments(how, thresh):
    """Test that how and thresh arguments cannot be both set."""
    with pytest.raises(TypeError):
        DropNA(axis=0, how=str(how), thresh=thresh)


@pytest.mark.parametrize("how", ["any", "all", "invalid_value"])
@pytest.mark.parametrize("thresh", [3, 0.9, True, False, "invalid_value", [1, 2, 3]])
def test_dropna_invalid_arguments(how, thresh):
    """Test that invalid arguments or combinations thereof are not accepted."""
    with pytest.raises((TypeError, ValueError)):
        DropNA(axis=0, how=str(how), thresh=thresh)
