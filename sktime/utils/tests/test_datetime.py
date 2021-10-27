# -*- coding: utf-8 -*-
"""Tests for datetime functions."""

__author__ = ["xiaobenbenecho"]

import pandas as pd
import numpy as np
import datetime

from sktime.utils.datetime import _get_freq


def test_get_freq():
    """Test whether get_freq runs without error."""
    x = pd.Series(
        index=pd.date_range(start="2017-01-01", periods=700, freq="W"),
        data=np.random.randn(700),
    )
    x1 = x.index
    x2 = x.resample("W").sum().index
    x3 = pd.Series(
        index=[
            datetime.datetime(2017, 1, 1) + datetime.timedelta(days=int(i))
            for i in np.arange(1, 100, 7)
        ]
    ).index
    x4 = [
        datetime.datetime(2017, 1, 1) + datetime.timedelta(days=int(i))
        for i in np.arange(1, 100, 7)
    ]
    assert _get_freq(x1) == "W"
    assert _get_freq(x2) == "W"
    assert _get_freq(x3) is None
    assert _get_freq(x4) is None
