#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for ClearSky transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np

from sktime.transformations.series.clear_sky import ClearSky
from sktime.datasets import load_solar


def test_clearsky_trafo_vals():
    """Tests the returned length of trafo with and without missing values."""
    y = load_solar(start="2021-05-01", end="2021-05-07")

    cs_model = ClearSky()
    y_trafo = cs_model.fit_transform(y)

    y_missing = y.copy()
    y_missing.iloc[48:96] = np.nan
    cs_model_missing = ClearSky()
    y_trafo_missing = cs_model_missing.fit_transform(y_missing)

    msg = "ClearSky transformer not returning correct number of values"
    assert y.count() == y_trafo.count(), msg
    assert y_missing.count() == y_trafo_missing.count(), msg
