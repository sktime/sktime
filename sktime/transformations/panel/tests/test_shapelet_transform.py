# -*- coding: utf-8 -*-
"""ShapeletTransform test code."""
import numpy as np
from numpy import testing

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform


def test_st_on_unit_test():
    """Test of ShapeletTransform on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(
        data, shapelet_transform_unit_test_data, decimal=4
    )


def test_st_on_basic_motions():
    """Test of ShapeletTransform on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(
        data, shapelet_transform_basic_motions_data, decimal=4
    )


shapelet_transform_unit_test_data = np.array(
    [
        [0.0845, 0.1536, 0.1812],
        [0.1006, 0.1373, 0.1423],
        [0.1557, 0.2457, 0.2811],
        [0.1437, 0.1117, 0.0832],
        [0.059, 0.117, 0.1461],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [0.987, 0.9232, 1.5005, 1.9604, 1.93, 1.629, 1.2492, 1.006],
        [1.0555, 1.0586, 0.6147, 0.9763, 0.5589, 1.0327, 1.0834, 1.1117],
        [1.9684, 1.7907, 2.4537, 1.9677, 2.0294, 1.3484, 1.9878, 0.5488],
        [1.0066, 1.1597, 1.0618, 0.323, 1.145, 1.0387, 0.6769, 1.0157],
        [0.2871, 0.0919, 1.7344, 1.9065, 1.73, 1.6298, 1.566, 1.1203],
    ]
)
