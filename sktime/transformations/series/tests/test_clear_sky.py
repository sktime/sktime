#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for ClearSky transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np

from sktime.datasets import load_solar
from sktime.transformations.series.clear_sky import ClearSky

output_chk = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.052,
    0.256,
    0.469,
    0.585,
    0.686,
    0.764,
    0.784,
    0.783,
    0.844,
    0.873,
    0.833,
    0.757,
    0.725,
    0.737,
    0.739,
    0.836,
    0.819,
    0.799,
    0.795,
    0.758,
    0.701,
    0.641,
    0.618,
    0.635,
    0.696,
    0.621,
    0.557,
    0.477,
    0.283,
    0.072,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


def test_clearsky_trafo_vals():
    """Tests clear sky trafo with and without missing values."""
    y = load_solar(start="2021-05-01", end="2021-05-07")

    cs_model = ClearSky()
    y_trafo = cs_model.fit_transform(y)

    msg = "ClearSky not transforming values consistently with stored values."
    assert np.all(output_chk == y_trafo[0:48].round(3).tolist()), msg

    y_missing = y.copy()
    y_missing.iloc[48:96] = np.nan
    cs_model_missing = ClearSky()
    y_trafo_missing = cs_model_missing.fit_transform(y_missing)

    msg = "ClearSky transformer not returning correct number of values."
    assert y.count() == y_trafo.count(), msg
    assert y_missing.count() == y_trafo_missing.count(), msg
