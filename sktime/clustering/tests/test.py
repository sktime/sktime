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
from sktime.clustering.tests.partition_based.test_ts_affi_prop import (
    test_time_series_affinity_propagation,
)
from sktime.clustering.tests.density_based.test_ts_ms import (
    test_time_series_mean_shift,
)
from sktime.clustering.tests.graph_theory_based.test_ts_spec_clus import (
    test_time_series_spectral_clustering,
)
from sktime.clustering.tests.graph_theory_based.test_ts_spec_biclust import (
    test_time_series_spectral_biclustering,
)
from sktime.clustering.tests.graph_theory_based.test_ts_spec_coclust import (
    test_time_series_spectral_coclustering,
)
from sktime.clustering.tests.hierachical.test_ts_agglo_clust import (
    test_time_series_agglo_clust,
)
from sktime.clustering.tests.hierachical.test_ts_feat_agglo_clust import (
    test_time_series_feat_agglo_clust,
)
from sktime.clustering.tests.density_based.test_ts_dbscan import test_time_series_dbscan
from sktime.clustering.tests.density_based.test_ts_optics import test_time_series_optics
from sktime.clustering.tests.hierachical.test_ts_birch import test_time_series_birch

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
# print("=================Testing k means===================================")
test_time_series_kmeans(train_x, train_y)
# print("=================Testing affinity propagation======================")
test_time_series_affinity_propagation(train_x, train_y)
# print("=================Testing density based=============================")
test_time_series_mean_shift(train_x, train_y)
test_time_series_dbscan(train_x, train_y)
test_time_series_optics(train_x, train_y)
# print("=================Testing spectral clustering======================")
test_time_series_spectral_clustering(train_x, train_y)
test_time_series_spectral_biclustering(train_x, train_y)
test_time_series_spectral_coclustering(train_x, train_y)
# print("=================Testing hierachical clustering===================")
test_time_series_agglo_clust(train_x, train_y)
test_time_series_feat_agglo_clust(train_x, train_y)
test_time_series_birch(train_x, train_y)
