"""Tests for base change point detection class."""

import numpy as np
import pandas as pd
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
def test_init_y_attribute():
    """Test that _y attribute is initialized to None in __init__.

    Regression test for bug #9889, where __init__ set self._Y = None (uppercase Y)
    instead of self._y = None, causing self._y to be absent before fit is called.
    This affects detectors with fit_is_empty=True, which never set self._y in fit.
    """
    from sktime.detection.dummy import ZeroAnomalies

    detector = ZeroAnomalies()
    detector.fit(pd.Series([1, 2, 3, 4, 5]))

    assert hasattr(detector, "_y"), (
        "BaseDetector.__init__ must initialize self._y; "
        "found self._Y (uppercase) instead"
    )
    assert detector._y is None
