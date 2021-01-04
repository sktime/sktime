# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sktime.clustering.utils import (
    convert_df_to_sklearn_format,
    SkLearn_Data,
    Data_Frame,
)


def test_cluster(df_x: Data_Frame):
    sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    km = KMeans(
        n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
    )
    km.fit(sklearn_train_data)
