# -*- coding: utf-8 -*-
"""Debug code for open issues."""
import shutil

import numpy as np
import pandas as pd

from sktime.classification.dummy import DummyClassifier
from sktime.classification.feature_based import SignatureClassifier
from sktime.datasets import (
    load_from_tsfile,
    load_gunpoint,
    load_japanese_vowels,
    load_plaid,
    load_UCR_UEA_dataset,
    write_dataframe_to_tsfile,
)
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.panel.signature_based._signature_method import (
    SignatureTransformer,
)
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils.sampling import stratified_resample


def debug_load_uea_dataset(return_type=None, split=None):
    """See https://github.com/sktime/sktime/pull/3021."""
    from sktime.datasets import load_UCR_UEA_dataset

    # NO EXTRACT PATH
    print("Use case 1: baked in data set. Should load without downloading")
    X, y = load_UCR_UEA_dataset(name="ArrowHead", return_type=return_type, split=split)
    print(
        " Case 1 Arrowhead y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )
    print(
        "Use case 2: not baked in, not already downloaded: should download to "
        "local_data"
    )
    X, y = load_UCR_UEA_dataset(name="BeetleFly", return_type=return_type, split=split)
    print(
        " Case 2 BeetleFly y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )
    print(
        "Use case 3: not baked in, but already downloaded: should load from "
        "local_data"
    )
    X, y = load_UCR_UEA_dataset(name="BeetleFly", return_type=return_type, split=split)
    print(
        " Case 3 BeetleFly y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )
    # Absolute extract path given: C:/temp    or C:/Temp/ or C:\\Temp or C:\\Temp\\
    print("Use case 4: dataset name currently in path/")
    X, y = load_UCR_UEA_dataset(
        name="ArrowHead", extract_path="C:\\Temp", return_type=return_type, split=split
    )
    print(
        " Case 4 a. Arrowhead y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )
    X, y = load_UCR_UEA_dataset(
        name="ArrowHead",
        extract_path="C:\\Temp\\",
        return_type=return_type,
        split=split,
    )
    print(
        " Case 4 b. Arrowhead y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )
    X, y = load_UCR_UEA_dataset(
        name="ArrowHead", extract_path="C:/Temp", return_type=return_type, split=split
    )
    print(
        " Case 4 c. Arrowhead y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )
    X, y = load_UCR_UEA_dataset(
        name="ArrowHead", extract_path="C:/Temp/", return_type=return_type, split=split
    )
    print(
        " Case 4 d. Arrowhead y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )

    print("Use case 5: dataset name NOT path/ but path exists")
    X, y = load_UCR_UEA_dataset(
        name="BeetleFly", extract_path="C:\\Temp", return_type=return_type, split=split
    )
    print(
        " Case 5 BeetleFly y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )

    print("Use case 6: dataset name NOT path/ and path does not exist")
    X, y = load_UCR_UEA_dataset(
        name="BeetleFly",
        extract_path="C:\\Temp\\Temp",
        return_type=return_type,
        split=split,
    )
    print(
        " Case 6 BeetleFly y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )
    print("Use case 7: dataset name and path exists")
    X, y = load_UCR_UEA_dataset(
        name="BeetleFly",
        extract_path="C:\\Temp\\Temp",
        return_type=return_type,
        split=split,
    )
    print(
        " Case 7 BeetleFly y[0] =", y[0], " load type = ", type(X), " size = ", X.shape
    )
    print("Use case 8: relative path name given e.g. Temp or ./Temp")
    X, y = load_UCR_UEA_dataset(
        name="BeetleFly", extract_path="Temp", return_type=return_type, split=split
    )
    print(
        " Case 8 a BeetleFly y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )
    X, y = load_UCR_UEA_dataset(
        name="Chinatown", extract_path="./Temp", return_type=return_type, split=split
    )
    print(
        " Case 8 b Chinatown y[0] =",
        y[0],
        " load type = ",
        type(X),
        " size = ",
        X.shape,
    )


# debug_load_uea_dataset(split="TRAIN")
# debug_load_uea_dataset(split="TEST")
# debug_load_uea_dataset(return_type="numpy3d")


def debug_write_dataframe_to_ts_file(name, extract_path=None):
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


# debug_write_dataframe_to_ts_file("GunPoint")
# debug_write_dataframe_to_ts_file("JapaneseVowels")
def debug_load_and_save():
    """Issue?."""
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


from sktime.datasets._data_io import _load_provided_dataset


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


def debug_numba_stc_2397(type):
    """See https://github.com/sktime/sktime/issues/2397."""
    import warnings

    warnings.simplefilter("ignore", category=FutureWarning)
    from sklearn.model_selection import train_test_split

    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    from sktime.classification.sklearn import RotationForest

    # make fake data
    if type == "int32" or type == "int64":  # Ensure not all zeros
        data = pd.DataFrame(100 * np.random.random((500, 25))).astype(type)
    else:
        data = pd.DataFrame(np.random.random((500, 25))).astype(type)

    # reshape to input into Shapelet Classifier
    data4train = data.apply(
        lambda row: pd.Series({"time-series": pd.Series(row.values)}), axis=1
    )

    # make targets
    targets = pd.Series(250 * [1] + 250 * [0])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        data4train, targets, test_size=0.7, random_state=42
    )

    # train
    clf = ShapeletTransformClassifier(n_shapelet_samples=100)

    clf.fit(X_train, y_train)


from numba import njit


@njit(
    fastmath=True,
    cache=True,
)
def z_normalise_debug(X):
    """Numba z-normalisation function for a single time series."""
    print("HEREEEEE: type of X = ", type(X[0]), " x.dtype = ", X.dtype)
    std = np.std(X)
    if std > 0:
        X_n = (X - np.mean(X)) / std
    else:
        X_n = X - np.mean(X)
    return X_n


def debug_numba2_2397():
    """Docstring."""
    #    d = z_normalise_series(data)
    #    print(" normed = ", d)
    data = np.zeros(10, dtype="int64")
    print(
        f"{type(data[0])} mean = {np.mean(data)} type mean = {type(np.mean(data))},returned "
        f"type {type((data -np.mean(data))[0])} type of zeros = {type(np.zeros(10)[0])}"
    )
    #    d = z_normalise_debug(data)
    #    print(" normed = ", d)
    data[0] = 100
    d = z_normalise_debug(data)
    print(" normed = ", d)


def debug_callibration_2662():
    """Issue 2662 https://github.com/sktime/sktime/issues/2662."""
    import sklearn.calibration
    import sklearn.pipeline

    from sktime.datasets import load_arrow_head, load_basic_motions
    from sktime.transformations.panel import rocket
    from sktime.transformations.panel.padder import PaddingTransformer

    X, y = load_basic_motions(return_X_y=True)
    n_jobs = -1

    featurizer_rocket = rocket.MiniRocket(n_jobs=n_jobs)
    featurizer_rocket = rocket.Rocket(n_jobs=n_jobs)
    featurizer_rocket = rocket.MultiRocket(n_jobs=n_jobs)
    featurizer_rocket = rocket.MiniRocketMultivariate(n_jobs=n_jobs)
    featurizer_rocket = rocket.MultiRocketMultivariate(n_jobs=n_jobs)
    classifier = sklearn.ensemble.HistGradientBoostingClassifier(
        loss="categorical_crossentropy"
    )

    base_estimator = sklearn.pipeline.Pipeline(
        [
            ("featurizer_rocket", featurizer_rocket),
            ("classifier", classifier),
        ],
    )

    calibrated_model = sklearn.calibration.CalibratedClassifierCV(
        base_estimator,
        cv=2,
        n_jobs=n_jobs,
    )

    calibrated_model.fit(X, y)


def debug_signatures_2374():
    """See #2374 https://github.com/sktime/sktime/issues/2374.

    Note in metaestimators.pt seems to add a trailing zero, wtf?
            Xt = X
            for _, _, transform in self._iter():
                Xt = transform.transform(Xt)
            return Xt
    """
    ## Test dataframe
    #    trainX = _make_panel_X(n_instances=40,n_timepoints=100)
    #    trainY = np.random.randint(low=0,high=2,size=40)
    trainX, trainY = load_UCR_UEA_dataset(
        name="BasicMotions", return_type="nested_univ", split="TRAIN"
    )

    print("Input type ", type(trainX), " Shape ", trainX.shape)
    sig1 = SignatureTransformer()
    sigC = SignatureClassifier()
    sig1.fit(trainX)
    transX = sig1.transform(trainX)
    sigC.fit(trainX, trainY)
    trainP = sigC.predict(trainX)
    print("Type transX = ", type(transX), " shape transX = ", transX.shape)
    print("Type of entry = ", type(transX.iloc[0, 0]), " shape transX = ", transX.shape)
    print(" Predictions = ", trainP)


def debug_signatures2_2374():
    """See #2374 https://github.com/sktime/sktime/issues/2374."""
    trainX, trainY = load_UCR_UEA_dataset(
        name="BasicMotions", return_type="numpy3D", split="TRAIN"
    )
    print("Input type ", type(trainX), " Shape ", trainX.shape)
    sig1 = SignatureTransformer()
    sigC = SignatureClassifier()
    sig1.fit(trainX)
    transX = sig1.transform(trainX)
    sigC.fit(trainX, trainY)
    trainP = sigC.predict(trainX)
    print("Type transX = ", type(transX), " shape transX = ", transX.shape)
    print("Type of entry = ", type(transX.iloc[0, 0]), " shape transX = ", transX.shape)
    print(" Predictions = ", trainP)


if __name__ == "__main__":
    # 2662 is a problem with n_jobs in the ROCKET transformer
    # debug_callibration_2662()
    debug_numba_stc_2397("int32")
    debug_numba_stc_2397("float64")
    debug_numba_stc_2397("float32")
    debug_numba_stc_2397("int64")
    # debug_numba2_2397()
