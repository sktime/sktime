# -*- coding: utf-8 -*-
"""Debug code for open issues."""
import shutil

import numpy as np

from sktime.datasets import (
    load_from_tsfile,
    load_gunpoint,
    load_japanese_vowels,
    load_plaid,
    load_UCR_UEA_dataset,
    write_dataframe_to_tsfile,
)
from sktime.datasets._data_io import _load_provided_dataset


def _debug_knn_2774():
    """https://github.com/sktime/sktime/issues/2774."""
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.datasets import load_unit_test

    knn = KNeighborsTimeSeriesClassifier()
    trainX, trainy = load_unit_test()
    knn.fit(trainX, trainy)
    #    'kd_tree’,'ball_tree'
    knn = KNeighborsTimeSeriesClassifier(algorithm="kd_tree")
    knn.fit(trainX, trainy)


def _debug_threaded_tde_3788_numpy(n_cases=20, n_dims=20, series_length=20):
    from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
    from sktime.classification.hybrid import HIVECOTEV2

    trainX = np.random.rand(n_cases, n_dims, series_length)
    trainY = np.random.randint(0, 2, n_cases)
    print(trainY)
    tde = HIVECOTEV2(n_jobs=2, time_limit_in_minutes=1)
    print(" beginning fit numpy")
    tde.fit(trainX, trainY)
    print(" beginning predict numpy")
    preds = tde.predict(trainX)
    print(" end predict numpy")


def _debug_threaded_tde_3788_pandas():
    from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

    n_cases = 100
    n_dims = 10
    series_length = 100
    trainX = np.random.rand(n_cases, n_dims, series_length)
    from datatypes import convert_to

    pd_trainX = convert_to(trainX, to_type="nested_univ", as_scitype="Panel")
    trainY = np.random.randint(0, 2, n_cases)
    tde = TemporalDictionaryEnsemble(n_jobs=2, time_limit_in_minutes=1)
    print(" beginning fit pandas")
    tde.fit(pd_trainX, trainY)
    print(" beginning predict pandas")
    pred = tde.predict(pd_trainX)


def _debug_pf():
    """PF just broken."""
    from sktime import show_versions

    show_versions()
    from sktime.classification.distance_based import ProximityForest
    from sktime.datasets import load_unit_test

    pf = ProximityForest()
    trainX, trainy = load_unit_test()
    pf.fit(trainX, trainy)


if __name__ == "__main__":
    _debug_threaded_tde_3788_numpy()
#    _debug_threaded_tde_3788_pandas()
