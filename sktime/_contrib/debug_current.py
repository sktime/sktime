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


def debug_pf():
    """PF just broken."""
    from sktime import show_versions

    show_versions()
    from sktime.classification.distance_based import ProximityForest
    from sktime.datasets import load_unit_test

    pf = ProximityForest()
    trainX, trainy = load_unit_test()
    pf.fit(trainX, trainy)


if __name__ == "__main__":
    debug_pf()
