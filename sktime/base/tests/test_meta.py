# -*- coding: utf-8 -*-
"""Test _HeterogenousMetaEstimator."""

import math

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler

from sktime.base._meta import _get_all_permutations, _HeterogenousMetaEstimator
from sktime.forecasting.trend import STLForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.impute import Imputer

steps = [
    ("boxcox", TabularToSeriesAdaptor(PowerTransformer(method="box-cox"))),
    ("robust", TabularToSeriesAdaptor(RobustScaler())),
    ("minmax", TabularToSeriesAdaptor(MinMaxScaler((1, 10)))),
    ("jeo", TabularToSeriesAdaptor(PowerTransformer(method="yeo-johnson"))),
    ("forecaster", STLForecaster(sp=12)),
    ("imputer", Imputer()),
]


def test_get_all_permutations():
    """Test that creation of all possible permutations works."""
    permutations = _get_all_permutations(steps)
    assert len(permutations) == math.factorial(
        4
    )  # 4 pre-processing steps that can be permuted

    for p in permutations:
        assert isinstance(p, list)
        assert len(p) == len(steps)
        assert p[-1] == "imputer"
        assert p[-2] == "forecaster"


def test_steps_permutation():
    """Test that given permutation leads to correctly sorted steps."""
    meta = _HeterogenousMetaEstimator()
    permutation = ["minmax", "robust", "jeo", "boxcox", "forecaster", "imputer"]
    permutation_steps = meta._steps_permutation(steps=steps, permutation=permutation)
    for p, ps in zip(permutation, permutation_steps):
        assert p == ps[0]
