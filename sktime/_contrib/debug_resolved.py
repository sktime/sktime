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
