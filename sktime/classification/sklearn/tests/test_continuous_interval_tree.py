# -*- coding: utf-8 -*-
"""ContinuousIntervalTree test code."""
import numpy as np
import pytest

from sktime.classification.sklearn import ContinuousIntervalTree


def test_nan_values():
    """Test that ContinuousIntervalTree can handle NaN values."""
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(10, 3))
    X[0:3, 0] = np.nan
    y = np.zeros(10)
    y[:5] = 1

    clf = ContinuousIntervalTree()
    clf.fit(X, y)
    clf.predict(X)

    # check inf values still raise an error
    X[0:3, 0] = np.inf
    with pytest.raises(ValueError):
        clf.fit(X, y)
