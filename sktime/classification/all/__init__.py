# -*- coding: utf-8 -*-
"""List of all imports used in classification package."""
__author__ = ["mloning", "TonyBagnall"]
__all__ = [
    "ShapeletTransformClassifier",
    "ROCKETClassifier",
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
    "RandomIntervalSpectralForest",
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
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.datasets import load_arrow_head
from sktime.datasets import load_basic_motions
from sktime.datasets import load_gunpoint
from sktime.datasets import load_osuleaf
