# -*- coding: utf-8 -*-
import os
import sktime
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.clustering.sklearn_utils import (
    convert_df_to_sklearn_format,
    Data_Frame,
    SkLearn_Data,
    Numpy_Array,
)
from sklearn.cluster import KMeans


def test_create_sklearn_k_means(df_x: Data_Frame, df_y: Data_Frame):
    """
    Method that is used to convert the sktime dataframe into a format that
    can be passed into sklearn algorithms

    Parameters
    ----------
    df_x: sktime dataframe
        Sktime training dataframe
    df_y: sktime dataframe
        Sktime test dataframe
    """
    sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    km = KMeans(
        n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
    )
    km.fit(sklearn_train_data)
    # print(y_km)


def test_convert_df_to_learn_format(df: Data_Frame):
    """
    Method that is used to convert the sktime dataframe into a format that
    can be passed into sklearn algorithms

    Parameters
    ----------
    df: sktime dataframe
        Sktime dataframe to be converted into sklearn format
    """
    sklearn_data: SkLearn_Data = convert_df_to_sklearn_format(df)
    assert isinstance(sklearn_data, Numpy_Array)
    for arr in sklearn_data:
        assert isinstance(arr, Numpy_Array)


# print("=============== START ================")
DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)
test_convert_df_to_learn_format(train_x)
test_create_sklearn_k_means(train_x, train_y)
