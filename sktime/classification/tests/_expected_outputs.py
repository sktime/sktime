# -*- coding: utf-8 -*-
"""Dictionaries of expected outputs of classifier predict runs."""

import numpy as np

# predict_proba results on unit test data
unit_test_proba = dict()

# predict_proba results on basic motions data
basic_motions_proba = dict()


unit_test_proba["BOSSEnsemble"] = np.array(
    [
        [0.33333333, 0.66666667],
        [0.66666667, 0.33333333],
        [0, 1],
        [1, 0],
        [0.66666667, 0.33333333],
        [0.66666667, 0.33333333],
        [1, 0],
        [0, 1],
        [0.33333333, 0.66666667],
        [1, 0],
    ]
)

unit_test_proba["IndividualBOSS"] = np.array(
    [
        [0.33333333, 0.66666667],
        [0.66666667, 0.33333333],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [0.33333333, 0.66666667],
        [0.66666667, 0.33333333],
        [1, 0],
    ]
)

unit_test_proba["ColumnEnsembleClassifier"] = np.array(
    [
        [0.33333333, 0.66666667],
        [0.66666667, 0.33333333],
        [0, 1],
        [1, 0],
        [1, 0],
        [0.66666667, 0.33333333],
        [1, 0],
        [0, 1],
        [0.33333333, 0.66666667],
        [1, 0],
    ]
)

basic_motions_proba["ColumnEnsembleClassifier"] = np.array(
    [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0.16666667, 0.83333333],
        [0.83333333, 0.16666667, 0, 0],
        [0, 0, 1, 0],
        [0.16666667, 0.83333333, 0, 0],
        [0, 1, 0, 0],
    ]
)
