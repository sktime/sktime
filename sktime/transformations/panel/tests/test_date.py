#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of DateTimeFeatures functionality."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.panel.date import DateTimeFeatures
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.fixture()
def X_panel():
    """Create panel data of two time series using pd-multiindex mtype."""
    return _make_hierarchical(hierarchy_levels=(2,), min_timepoints=3, max_timepoints=3)


def test_transform_panel(X_panel):
    """Test `.transform()` on panel data."""
    transformer = DateTimeFeatures(
        manual_selection=["year", "month_of_year", "day_of_month"]
    )
    Xt = transformer.fit_transform(X_panel)

    expected = pd.DataFrame(
        index=X_panel.index,
        data={
            "c0": X_panel["c0"].values,
            "year": [2000, 2000, 2000, 2000, 2000, 2000],
            "month_of_year": [1, 1, 1, 1, 1, 1],
            "day_of_month": [1, 2, 3, 1, 2, 3],
        },
    )
    assert_frame_equal(Xt, expected)
