# -*- coding: utf-8 -*-
import os
import sktime
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.clustering.tests.test_utils import (
    test_convert_df_to_learn_format,
    test_check_shape,
)
from sktime.clustering.tests.test_cluster import test_cluster

# from sktime.clustering.tests.partition_based.test_ts_affi_prop import (
#     test_time_series_affinity_propagation,
# )
from sktime.clustering.tests.partition_based.test_time_series_kmeans import (
    test_time_series_kmeans,
)

# from sktime.clustering.tests.density_based.test_time_series_mean_shift import (
#     test_time_series_mean_shift,
# )
# from sktime.clustering.tests.graph_theory_based.test_ts_spec_clus import (
#     test_time_series_spectral_clustering,
# )

# print("=============== START ================")
DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)

# Utils testing
test_convert_df_to_learn_format(train_x)
test_check_shape()


train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)
# Basic cluster methods and classes
test_cluster()

# Specific clustering algorithms
test_time_series_kmeans(train_x, train_y)
# test_time_series_affinity_propagation(train_x, train_y)
# test_time_series_mean_shift(train_x, train_y)
# test_time_series_spectral_clustering(train_x, train_y)
