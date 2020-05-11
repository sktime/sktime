#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "test_estimator"
]

import pytest
from sktime.utils import all_estimators
from sktime.utils.testing.estimator_checks import check_estimator

# TODO fix estimators to pass all tests
EXCLUDED = [
    "BOSSEnsemble",
    "BOSSIndividual",
    "ElasticEnsemble",
    "KNeighborsTimeSeriesClassifier",
    "MrSEQLClassifier",
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
    "RandomIntervalSpectralForest",
    "ShapeletTransformClassifier",
    "TimeSeriesForestClassifier",
    "TimeSeriesForestRegressor"
]
from sktime.transformers.series_as_features.base import BaseTransformer
ALL_ESTIMATORS = [e[1] for e in all_estimators() if
                  e[0] not in EXCLUDED and
                  not issubclass(e[1], BaseTransformer)]


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_estimator(Estimator):
    # We run a number of basic checks on all estimators to ensure correct
    # implementation of our framework and compatibility with scikit-learn
    check_estimator(Estimator)
