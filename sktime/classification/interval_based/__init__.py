# -*- coding: utf-8 -*-
"""Interval based time series classifiers."""
__all__ = [
    "TimeSeriesForestClassifier",
    "RandomIntervalSpectralForest",
    "RandomIntervalSpectralEnsemble",
    "SupervisedTimeSeriesForest",
    "CanonicalIntervalForest",
    "DrCIF",
]

from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.interval_based._rise import (
    RandomIntervalSpectralForest,  # todo remove in 0.10.0
)
from sktime.classification.interval_based._rise import RandomIntervalSpectralEnsemble
from sktime.classification.interval_based._stsf import SupervisedTimeSeriesForest
from sktime.classification.interval_based._tsf import TimeSeriesForestClassifier
