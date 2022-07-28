# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseAligner API points."""

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datatypes._check import check_raise
from sktime.registry import all_estimators
from sktime.utils._testing.series import _make_series
from sktime.utils.validation._dependencies import _check_estimator_deps

# get all aligners
ALIGNERS = all_estimators(estimator_types="aligner", return_names=False)
INVALID_X_INPUT_TYPES = [list(), tuple()]
INVALID_y_INPUT_TYPES = [list(), tuple()]


@pytest.mark.parametrize("Aligner", ALIGNERS)
def test_get_alignment(Aligner):
    """Test that get_alignment returns an alignment (iloc)."""
    if not _check_estimator_deps(Aligner, severity="none"):
        return None

    f = Aligner.create_test_instance()

    X = [_make_series(n_columns=2), _make_series(n_columns=2)]
    align = f.fit(X).get_alignment()

    check_raise(align, mtype="alignment", scitype="Alignment")


@pytest.mark.parametrize("Aligner", ALIGNERS)
def test_get_alignment_loc(Aligner):
    """Test that get_alignment returns an alignment (loc)."""
    if not _check_estimator_deps(Aligner, severity="none"):
        return None

    f = Aligner.create_test_instance()

    X = [_make_series(n_columns=2), _make_series(n_columns=2)]
    align = f.fit(X).get_alignment_loc()

    check_raise(align, mtype="alignment_loc", scitype="Alignment")


@pytest.mark.parametrize("Aligner", ALIGNERS)
def test_get_aligned(Aligner):
    """Test that get_aligned returns list of series with same columns."""
    if not _check_estimator_deps(Aligner, severity="none"):
        return None

    f = Aligner.create_test_instance()

    X = [_make_series(n_columns=2), _make_series(n_columns=2)]
    n = len(X)
    X_aligned = f.fit(X).get_aligned()

    msg = f"{Aligner.__name__}.get_aligned must return list of pd.DataFrame"
    msg += ", same length as X in fit"
    col_msg = f"{Aligner.__name__}.get_aligned series must have same columns as in X"
    assert isinstance(X_aligned, list), msg
    assert len(X_aligned) == n, msg

    for i in range(n):
        Xi = X_aligned[i]
        assert isinstance(Xi, pd.DataFrame), msg
        assert set(Xi.columns) == set(X[i].columns), col_msg
