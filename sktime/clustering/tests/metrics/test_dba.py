# -*- coding: utf-8 -*-

import numpy as np

from sktime.clustering.metrics.averaging import dba
from sktime.clustering.metrics.medoids import medoids
from sktime.datasets import load_acsf1
from sktime.datatypes import convert_to
from sktime.distances.tests._utils import create_test_distance_numpy


def test_dba():
    """Test medoids."""
    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")
    # X_train = create_test_distance_numpy(10, 4, 3, random_state=2)

    X_train = convert_to(X_train, "numpy3D")

    X_train = X_train[:5]

    test_dba = dba(X_train)
    test_medoids = medoids(X_train)
    joe = ""
