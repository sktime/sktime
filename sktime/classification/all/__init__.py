# -*- coding: utf-8 -*-
"""All time series classifiers."""

__author__ = ["Markus Löning"]
__all__ = [
    "ShapeletTransformClassifier",
    "MrSEQLClassifier",
    "RocketClassifier",
    "BOSSEnsemble",
    "IndividualBOSS",
    "TemporalDictionaryEnsemble",
    "IndividualTDE",
    "KNeighborsTimeSeriesClassifier",
    "ProximityStump",
    "ProximityTree",
    "ProximityForest",
    "ElasticEnsemble",
    "TimeSeriesForestClassifier",
    "RandomIntervalSpectralEnsemble",
    "SupervisedTimeSeriesForest",
    "ComposableTimeSeriesForestClassifier",
    "ColumnEnsembleClassifier",
    "pd",
    "np",
    "load_gunpoint",
    "load_osuleaf",
    "load_basic_motions",
    "load_arrow_head",
]

import numpy as np
import pandas as pd

from sktime.classification.compose import (
    ColumnEnsembleClassifier,
    ComposableTimeSeriesForestClassifier,
)
from sktime.classification.dictionary_based import (
    BOSSEnsemble,
    IndividualBOSS,
    IndividualTDE,
    TemporalDictionaryEnsemble,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
    ProximityForest,
    ProximityStump,
    ProximityTree,
)
from sktime.classification.interval_based import (
    RandomIntervalSpectralEnsemble,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import (
    MrSEQLClassifier,
    ShapeletTransformClassifier,
)
from sktime.datasets import (
    load_arrow_head,
    load_basic_motions,
    load_gunpoint,
    load_osuleaf,
)
