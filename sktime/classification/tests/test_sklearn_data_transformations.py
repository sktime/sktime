# -*- coding: utf-8 -*-
"""Unit tests for classifier compatability with sklearn data transformations."""

__author__ = ["MatthewMiddlehurst"]

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from sktime.classification.interval_based import CanonicalIntervalForest

DATA_ARGS = [
    {"return_numpy": True, "n_columns": 2},
    {"return_numpy": False, "n_columns": 2},
]
COMPOSITE_ESTIMATORS = [
    Pipeline(
        [
            ("pca", PCA()),
            (
                "clf",
                CanonicalIntervalForest(
                    n_estimators=3, n_intervals=2, att_subsample_size=2
                ),
            ),
        ]
    ),
    VotingClassifier(
        estimators=[
            (
                "clf1",
                CanonicalIntervalForest(
                    n_estimators=3, n_intervals=2, att_subsample_size=2
                ),
            ),
            (
                "clf2",
                CanonicalIntervalForest(
                    n_estimators=3, n_intervals=2, att_subsample_size=2
                ),
            ),
            (
                "clf3",
                CanonicalIntervalForest(
                    n_estimators=3, n_intervals=2, att_subsample_size=2
                ),
            ),
        ]
    ),
    CalibratedClassifierCV(
        base_estimator=CanonicalIntervalForest(
            n_estimators=3, n_intervals=2, att_subsample_size=2
        ),
        cv=3,
    ),
]
