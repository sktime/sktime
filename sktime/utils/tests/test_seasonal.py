# -*- coding: utf-8 -*-
"""Testing seasonal utilities."""

import numpy as np
import pytest

from sktime.utils._testing.series import _make_series
from sktime.utils.seasonality import _pivot_sp, _unpivot_sp


@pytest.mark.parametrize("n_timepoints", [50, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
def test_pivot_sp(sp, index_type, n_timepoints):
    """Test that random_partition returns a disjoint partition."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )

    df_pivot = _pivot_sp(df, sp)
