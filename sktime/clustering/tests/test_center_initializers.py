# -*- coding: utf-8 -*-
import os

import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.clustering._center_initializers import RandomCenterInitializer

DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
X, Y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)


def test_random_cluster_center_initializer():
    random_clusters = RandomCenterInitializer(X, 10)
    random_clusters.initialize_centers()
