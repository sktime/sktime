"""Compositions for classifiers."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = [
    "ClassifierPipeline",
    "ComposableTimeSeriesForestClassifier",
    "ColumnEnsembleClassifier",
    "MultiplexClassifier",
    "SklearnClassifierPipeline",
    "WeightedEnsembleClassifier",
]

from sktime.classification.compose._column_ensemble import ColumnEnsembleClassifier
from sktime.classification.compose._multiplexer import MultiplexClassifier
from sktime.classification.compose._pipeline import (
    ClassifierPipeline,
    SklearnClassifierPipeline,
)
