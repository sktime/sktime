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


def debug_write_dataframe_to_ts_file_3499(name, extract_path=None):
    """See https://github.com/sktime/sktime/issues/3499."""
    from sktime.datatypes import check_is_scitype

    #    X, y = load_UCR_UEA_dataset(name=name, extract_path="C:\\Temp",
    #                                return_type="numpy3D", split="TRAIN")
    #    print(" series shape  = ",X.shape)
    X, y = load_UCR_UEA_dataset(name=name, extract_path=extract_path)
    X_valid, _, X_metadata = check_is_scitype(X, scitype="Panel", return_metadata=True)
    print(X_metadata)
    series_length = X.iloc[0, 0].size
    print(" series length  = ", series_length)
    write_dataframe_to_tsfile(
        X,
        "C:\\Temp\\WriteTest",
        class_value_list=y,
        equal_length=X_metadata["is_equal_length"],
        problem_name=name,
    )


def debug_load_and_save_3499():
    """See https://github.com/sktime/sktime/issues/3499."""
    X1, y1 = load_gunpoint()
    write_dataframe_to_tsfile(
        X1,
        "C:\\Temp\\WriteTest",
        class_value_list=y1,
        equal_length=True,
        problem_name="GunPoint",
    )
    X2, y2 = load_from_tsfile(
        full_file_path_and_name="C:\\Temp\\WriteTest\\GunPoint" "\\GunPoint.ts"
    )
    assert np.array_equal(y1, y2)

    X1, y1 = load_japanese_vowels()
    print("Type of y1 = ", type(y1))
    write_dataframe_to_tsfile(
        X1,
        "C:\\Temp\\WriteTest",
        class_value_list=y1,
        equal_length=False,
        problem_name="JapaneseVowels",
    )
    X2, y2 = load_from_tsfile(
        full_file_path_and_name="C:\\Temp\\WriteTest\\JapaneseVowels"
        "\\JapaneseVowels.ts"
    )
    assert np.array_equal(y1, y2)

    X1, y1 = load_plaid()
    print("Type of y1 = ", type(y1))
    write_dataframe_to_tsfile(
        X1,
        "C:\\Temp\\WriteTest",
        class_value_list=y1,
        equal_length=False,
        problem_name="Fred",
    )
    X2, y2 = load_from_tsfile(
        full_file_path_and_name="C:\\Temp\\WriteTest\\Fred" "\\Fred.ts"
    )
    assert np.array_equal(y1, y2)


def debug_testing_load_and_save_3499():
    """Test load and save, related to https://github.com/sktime/sktime/issues/3499."""
    from datasets import write_panel_to_tsfile

    return_type = "nested_univ"
    dataset_name = "ItalyPowerDemand"
    X, y = _load_provided_dataset(dataset_name, split="TRAIN", return_type=return_type)
    write_panel_to_tsfile(data=X, path="./Temp", target=y, problem_name=dataset_name)
    load_path = f"./Temp/{dataset_name}/{dataset_name}.ts"
    newX, newy = load_from_tsfile(
        full_file_path_and_name=load_path, return_data_type=return_type
    )
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


## https://github.com/sktime/sktime/issues/2774
def debug_knn_2774():
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.datasets import load_unit_test
    knn = KNeighborsTimeSeriesClassifier()
    trainX, trainy = load_unit_test()
    knn.fit(trainX, trainy)
    #    'kd_tree’,'ball_tree'
    knn = KNeighborsTimeSeriesClassifier(algorithm="kd_tree")
    knn.fit(trainX, trainy)


if __name__ == "__main__":
    debug_knn_2774()
