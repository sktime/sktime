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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_y_initialized_in_init():
    """Test that _y attribute is initialized to None in __init__.

    Regression test for bug #9889, where __init__ set self._Y = None (uppercase Y)
    instead of self._y = None. For detectors with fit_is_empty=True, fit() returns
    early without ever setting self._y, leaving self._y absent from the object.

    This checks the invariant directly on a fresh (unfitted) instance, which is
    exactly where the bug manifested: the missing initialization in __init__.
    """
    from sktime.detection.dummy import ZeroAnomalies

    detector = ZeroAnomalies()

    assert hasattr(detector, "_y"), (
        "BaseDetector.__init__ must initialize self._y to None; "
        "found self._Y (uppercase) was set instead, leaving self._y absent"
    )
    assert detector._y is None
