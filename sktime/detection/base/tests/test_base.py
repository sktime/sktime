"""Tests for base change point detection class."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_fit_transform_numpy():
    """Test that numpy input works for fit_transform.

    Failure case of bug #8325.
    """
    from sktime.detection.lof import SubLOF

    data = np.array([0, 0.5, 2, 0.1, 0, 0, 0, 2, 0, 0, 0.3, -1, 0, 2, 0.2])
    model = SubLOF(3, window_size=5, novelty=True)
    pred = model.fit_transform(data)

    assert pred.shape[0] == data.shape[0]

    # also test fit alone
    model = SubLOF(3, window_size=5, novelty=True)
    model.fit(data)
