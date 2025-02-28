"""Tests for KNeighborsTimeSeriesRegressor."""

import numpy as np
import pandas as pd
import pytest

from sktime.regression.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesRegressor,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(KNeighborsTimeSeriesRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_knn_kneighbors():
    """Tests kneighbors method and absence of bug #3798."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    Xtrain = _make_hierarchical(hierarchy_levels=(3,), n_columns=3)
    Xtest = _make_hierarchical(hierarchy_levels=(5,), n_columns=3)

    ytrain = pd.Series([1, 1.5, 2])

    kntsc = KNeighborsTimeSeriesRegressor(n_neighbors=1)
    kntsc.fit(Xtrain, ytrain)

    ret = kntsc.kneighbors(Xtest)
    assert isinstance(ret, tuple)
    assert len(ret) == 2

    dist, ind = ret
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (5, 1)
    assert isinstance(ind, np.ndarray)
    assert ind.shape == (5, 1)
