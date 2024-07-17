"""ShapeletTransformPyts test code."""

import numpy as np
import pytest
from numpy import testing

from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.shapelet_transform import ShapeletTransformPyts


@pytest.mark.skipif(
    not run_test_for_class(ShapeletTransformPyts),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_stp_on_unit_test():
    """Test of ShapeletTransformPyts on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    stp = ShapeletTransformPyts(n_shapelets=5, random_state=0)
    stp.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = stp.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(
        data, shapelet_transform_unit_test_data, decimal=4
    )


shapelet_transform_unit_test_data = np.array(
    [
        [176.0, 14.866069, 17.262677, 0.000000, 7.0],
        [0.0, 51.908573, 82.197932, 65.517173, 15.0],
        [223.0, 30.413813, 0.000000, 17.262677, 7.0],
        [1.0, 61.159627, 71.515733, 59.502101, 19.0],
        [32.0, 0.000000, 30.413813, 14.866069, 0.0],
    ]
)
