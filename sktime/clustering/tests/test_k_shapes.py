# -*- coding: utf-8 -*-
"""Tests for time series k-shapes."""
# from sklearn import metrics

from sktime.clustering._k_shapes import KShapes
from sktime.datasets import load_basic_motions

expected_results = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    1,
    3,
    3,
    3,
    1,
    1,
    1,
    3,
    1,
    1,
    3,
    6,
    6,
    6,
    6,
    6,
    3,
    6,
    6,
    6,
    1,
    6,
    1,
    1,
    6,
    6,
    6,
    1,
    7,
    6,
]

inertia = 0.4996949455654395

expected_score = 0.8192307692307692

expected_iters = 0

expected_labels = [
    7,
    4,
    6,
    4,
    4,
    4,
    6,
    4,
    4,
    4,
    1,
    1,
    3,
    3,
    1,
    1,
    1,
    1,
    3,
    3,
    6,
    6,
    2,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    1,
    1,
    6,
]


def test_kshapes():
    """Test implementation of Kshapes."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kshapes = KShapes(random_state=1, n_clusters=3, verbose=True)
    kshapes.fit(X_train)
    # test_shape_result = kshapes.predict(X_test)
    # score = metrics.rand_score(y_test, test_shape_result)
    # proba = kshapes.predict_proba(X_test)

    # assert np.array_equal(test_shape_result, expected_results)
    # assert score == expected_score
    # assert kshapes.n_iter_ == expected_iters
    # assert np.array_equal(kshapes.labels_, expected_labels)
    # assert isinstance(kshapes.cluster_centers_, np.ndarray)
    # assert proba.shape == (40, 8)
    #
    # for val in proba:
    #     assert np.count_nonzero(val == 1.0) == 1
