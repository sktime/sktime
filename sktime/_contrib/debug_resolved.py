# -*- coding: utf-8 -*-
"""Debug code for resolved issues."""


def debug_concat():
    """See https://github.com/sktime/sktime/pull/3546."""
    trainX = _make_panel_X(n_instances=40)
    trainY = np.random.randint(low=0, high=2, size=40)
    dummy = DummyClassifier()
    # this works
    dummy.fit(trainX, trainY)
    trainX2 = _make_panel_X(n_instances=40)
    trainX2 = pd.concat([trainX2, trainX])
    trainY2 = np.random.randint(low=0, high=2, size=80)
    # this throws a type error
    #    trainX2 = trainX2.reset_index(drop=True)
    X, y, X2, y2 = stratified_resample(trainX, trainY, trainX2, trainY2, 0)
    dummy.fit(X, y)
    from datasets import load_unit_test

    trainX, trainy = load_unit_test(split="TRAIN")
    testX, testy = load_unit_test(split="TEST")
    new_trainX, new_trainy, new_testX, new_testy = stratified_resample(
        trainX, trainy, testX, testy, 0
    )
    # count class occurrences
    unique_train, counts_train = np.unique(trainy, return_counts=True)
    unique_test, counts_test = np.unique(testy, return_counts=True)
    unique_train_new, counts_train_new = np.unique(new_trainy, return_counts=True)
    unique_test_new, counts_test_new = np.unique(new_testy, return_counts=True)
    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


def debug_random_shapelet_transform():
    """See https://github.com/sktime/sktime/pull/3564."""
    trainX = _make_panel_X(n_instances=10)
    trainY1 = np.random.randint(low=0, high=5, size=10)
    trainY2 = np.random.randint(low=0, high=20, size=10)
    trainY3 = np.random.random(size=10)
    st = RandomShapeletTransform(batch_size=20, max_shapelets=5, n_shapelet_samples=50)
    st.fit(trainX, trainY2)


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
    """Numba z-normalisation function for a single time series.

    numba with STC https://github.com/sktime/sktime/issues/2397.
    """
    print("HEREEEEE: type of X = ", type(X[0]), " x.dtype = ", X.dtype)
    std = np.std(X)
    if std > 0:
        X_n = (X - np.mean(X)) / std
    else:
        X_n = X - np.mean(X)
    return X_n


def debug_numba2_2397():
    """Numba with STC https://github.com/sktime/sktime/issues/2397."""
    #    d = z_normalise_series(data)
    #    print(" normed = ", d)
    data = np.zeros(10, dtype="int64")
    print(
        f"{type(data[0])} mean = {np.mean(data)} type mean = {type(np.mean(data))},returned "
        f"type {type((data - np.mean(data))[0])} type of zeros = {type(np.zeros(10)[0])}"
    )
    #    d = z_normalise_debug(data)
    #    print(" normed = ", d)
    data[0] = 100
    d = z_normalise_debug(data)
    print(" normed = ", d)


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


if __name__ == "__main__":
    # 2662 is a problem with n_jobs in the ROCKET transformer
    # debug_write_dataframe_to_ts_file("GunPoint")
    # debug_write_dataframe_to_ts_file("JapaneseVowels")
    debug_numba_stc_2397("int32")
    debug_numba_stc_2397("float64")
    debug_numba_stc_2397("float32")
    debug_numba_stc_2397("int64")
    # debug_numba2_2397()
