"""ShapeletTransform test code."""

import numpy as np
import pytest
from numpy import testing

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform


@pytest.mark.xfail(reason="known failure that needs investigation, see issue #7725")
@pytest.mark.skipif(
    not run_test_for_class(RandomShapeletTransform),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.xfail(reason="known sporadic failure, likely pseudo-random instability")
@pytest.mark.skipif(
    not run_test_for_class(RandomShapeletTransform),
    reason="run test only if softdeps are present and incrementally (if requested)",
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
        [0.0300, 0.0001, 0.00571, 0.0430],
        [0.0723, 0.0500, 0.0412, 0.0001],
        [0.0001, 0.3001, 0.1120, 0.0656],
        [0.1437, 0.1040, 0.0001, 0.0140],
        [0.0586, 0.1160, 0.05889, 0.0375],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [0.0, 1.5154, 2.0713, 1.6964, 1.93, 1.7166, 1.6446, 1.3747],
        [2.1278, 1.7971, 0.0, 1.5443, 0.0, 2.0929, 2.0797, 2.4141],
        [1.8443, 1.6338, 2.2822, 1.9677, 2.0294, 0.0, 1.9878, 0.0],
        [1.8894, 1.8463, 2.1423, 0.0, 1.8901, 2.1195, 0.0, 1.8769],
        [1.6369, 0.0, 1.9507, 1.8272, 1.73, 2.0308, 1.9637, 1.9896],
    ]
)
