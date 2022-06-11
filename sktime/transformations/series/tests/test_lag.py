# -*- coding: utf-8 -*-
"""Tests for Lag transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.datatypes import get_examples
from sktime.transformations.series.lag import Lag
from sktime.utils._testing.series import _make_series

# some examples with range vs time index, univariate vs multivariate (mv)
X_range_idx = get_examples("pd.DataFrame")[0]
X_range_idx_mv = get_examples("pd.DataFrame")[1]
X_time_idx = _make_series()
X_time_idx_mv = _make_series(n_columns=2)

# all fixtures
X_fixtures = [X_range_idx, X_range_idx_mv, X_time_idx, X_time_idx_mv]

# fixtures with time index
X_time_fixtures = [X_time_idx, X_time_idx_mv]

@pytest.mark.parametrize("theta", [(1, 1.5), (0, 1, 2), (0.25, 0.5, 0.75, 1, 2)])
def test_thetalines_shape(theta):
    y = load_airline()
    t = ThetaLinesTransformer(theta)
    t.fit(y)
    actual = t.transform(y)
    assert actual.shape == (y.shape[0], len(theta))
