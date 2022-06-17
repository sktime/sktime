# -*- coding: utf-8 -*-
"""Tests for the subsetting transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.transformations.series.subset import ColumnSelect, IndexSubset


@pytest.mark.parametrize("index_treatment", ["keep", "remove"])
def test_indexsubset_indextreatment(index_treatment):
    """Test that index_treatment behaviour in IndexSubsetworks as intended."""
    X = load_airline()[0:32]
    y = load_airline()[24:42]
    transformer = IndexSubset(index_treatment=index_treatment)
    X_subset = transformer.fit_transform(X=X, y=y)
    if index_treatment == "remove":
        assert X_subset.index == X.index.intersect(y.index)
    elif index_treatment == "keep":
        assert X_subset.index == y.index


@pytest.mark.parametrize("index_treatment", ["keep", "remove"])
def test_columnselect_indextreatment(index_treatment):
    """Test that index_treatment behaviour in ColumnSelect works as intended."""
    X = load_longley()[1]
    columns = ["GNPDEFL", "POP", "FOO"]
    transformer = ColumnSelect(columns=columns, index_treatment=index_treatment)
    X_subset = transformer.fit_transform(X=X)
    if index_treatment == "remove":
        assert X_subset.columns == X.index.intersect(columns)
    elif index_treatment == "keep":
        assert X_subset.columns == pd.Index(columns)


def test_columnselect_int():
    """Test that integer/iloc subsetting in ColumnSelect works as intended."""
    X = load_longley()[1]
    columns = [0, 2, 4, 10]
    transformer = ColumnSelect(columns=columns)
    X_subset = transformer.fit_transform(X=X)

    assert X_subset.columns == X.columns[[0, 2, 4]]
