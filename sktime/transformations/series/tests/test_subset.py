"""Tests for the subsetting transformers."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.subset import ColumnSelect, IndexSubset


@pytest.mark.skipif(
    not run_test_for_class(IndexSubset),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("index_treatment", ["keep", "remove"])
def test_indexsubset_indextreatment(index_treatment):
    """Test that index_treatment behaviour in IndexSubsetworks as intended."""
    X = load_airline()[0:32]
    y = load_airline()[24:42]
    transformer = IndexSubset(index_treatment=index_treatment)
    X_subset = transformer.fit_transform(X=X, y=y)
    if index_treatment == "remove":
        assert X_subset.index.equals(X.index.intersection(y.index))
    elif index_treatment == "keep":
        assert X_subset.index.equals(y.index)


@pytest.mark.skipif(
    not run_test_for_class(ColumnSelect),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("index_treatment", ["keep", "remove"])
def test_columnselect_indextreatment(index_treatment):
    """Test that index_treatment behaviour in ColumnSelect works as intended."""
    X = load_longley()[1]
    columns = ["GNPDEFL", "POP", "FOO"]
    transformer = ColumnSelect(columns=columns, index_treatment=index_treatment)
    X_subset = transformer.fit_transform(X=X)

    columns_idx = pd.Index(columns)
    in_cols = columns_idx.isin(X.columns)
    col_X_and_cols = columns_idx[in_cols]

    if index_treatment == "remove":
        assert X_subset.columns.equals(col_X_and_cols)
    elif index_treatment == "keep":
        assert X_subset.columns.equals(columns_idx)


@pytest.mark.skipif(
    not run_test_for_class(ColumnSelect),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_columnselect_int():
    """Test that integer/iloc subsetting in ColumnSelect works as intended."""
    X = load_longley()[1]
    columns = [0, 2, 4, 10]
    transformer = ColumnSelect(columns=columns)
    X_subset = transformer.fit_transform(X=X)

    assert X_subset.columns.equals(X.columns[[0, 2, 4]])


@pytest.mark.skipif(
    not run_test_for_class(ColumnSelect),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_columnselect_as_first_step_in_transformedtargetforecaster():
    y = load_longley()[1][["GNP", "UNEMP"]]
    fc = ColumnSelect(["GNP"]) * NaiveForecaster()
    fc.fit(y)
    fc.predict(fh=[1])
