#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel"]
__all__ = []

import pandas as pd
from sktime.datasets import load_uschange


def test_uschange():
    X, y = load_uschange()

    X_test = pd.read_pickle('uschange_X.pkl')
    y_test = pd.read_pickle('uschange_y.pkl')

    pd.testing.assert_frame_equal(X, X_test)
    pd.testing.assert_series_equal(y, y_test)
