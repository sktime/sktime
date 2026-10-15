# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for DetectorAsTransformer."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.compose import DetectorAsTransformer
from sktime.tests.test_switch import run_test_for_class


def _X(n_columns):
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    data = {
        name: np.arange(40, dtype=float) * (i + 1)
        for i, name in enumerate(list("abcdefg")[:n_columns])
    }
    return pd.DataFrame(data, index=index)


@pytest.mark.skipif(
    not run_test_for_class(DetectorAsTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_input():
    """Test multivariate X, see the capability:multivariate tag.

    ``X_inner_mtype`` is ``pd.Series``, which is univariate by construction, so
    multivariate ``X`` can only be handled by vectorizing over columns. The tag
    claimed multivariate support directly, and multivariate input raised
    ``ValueError: input must be univariate pd.DataFrame, with one column``.
    """
    X = _X(2)

    Xt = DetectorAsTransformer.create_test_instance().fit_transform(X)

    assert len(Xt) == len(X)
    # one output column per input column, rather than a single collapsed one
    assert Xt.shape[1] == X.shape[1]


@pytest.mark.skipif(
    not run_test_for_class(DetectorAsTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_univariate_input_unchanged():
    """Test that univariate X is unaffected by the vectorization."""
    X = _X(1)

    Xt = DetectorAsTransformer.create_test_instance().fit_transform(X)

    assert Xt.shape == (len(X), 1)


@pytest.mark.skipif(
    not run_test_for_class(DetectorAsTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_matches_by_column():
    """Test that the vectorized result equals applying the detector per column."""
    X = _X(2)

    Xt = DetectorAsTransformer.create_test_instance().fit_transform(X)

    for i, col in enumerate(X.columns):
        expected = DetectorAsTransformer.create_test_instance().fit_transform(X[[col]])
        np.testing.assert_array_equal(
            np.asarray(Xt.iloc[:, i]).ravel(), np.asarray(expected).ravel()
        )
