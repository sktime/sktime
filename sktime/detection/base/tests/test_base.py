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
    from sktime.detection.dummy import DummyRegularAnomalies

    data = np.array([0, 0.5, 2, 0.1, 0, 0, 0, 2, 0, 0, 0.3, -1, 0, 2, 0.2])
    model = DummyRegularAnomalies()
    pred = model.fit_transform(data)

    assert pred.shape[0] == data.shape[0]

    # also test fit alone
    model = DummyRegularAnomalies()
    model.fit(data)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
@pytest.mark.parametrize("input_type", ["DataFrame", "Series", "ndarray"])
def test_update_after_fit_supported_input_types(input_type):
    """Test that update succeeds for all input formats accepted by fit.

    Regression test for bug #11606: BaseDetector.update failed after fitting
    supported Series or NumPy inputs, because fit stored the raw input X in
    self._X while update expected a converted X_inner (pd.DataFrame).

    For DataFrame: update should pool old and new data correctly.
    For Series/ndarray: update should not raise a ValueError or AttributeError.
    """
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.Series(np.arange(15, dtype=float))

    if input_type == "DataFrame":
        X_fit = X.iloc[:10].to_frame()
        X_update = X.iloc[10:].to_frame()
    elif input_type == "Series":
        X_fit = X.iloc[:10]
        X_update = X.iloc[10:]
    elif input_type == "ndarray":
        X_fit = X.iloc[:10].to_numpy()
        X_update = X.iloc[10:].to_numpy()

    detector = DummyRegularAnomalies()
    detector.fit(X_fit)
    detector.update(X_update)

    assert len(detector._X) == 15
    np.testing.assert_array_equal(
        detector._X.to_numpy().flatten(), np.arange(15, dtype=float)
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_with_y_after_fit_without_y():
    """Test that update with labels after an unlabelled fit does not raise TypeError.

    Regression test for bug #11606: calling update(X, y=labels) after fit(X)
    (without y) was raising TypeError because fit stored self._y = None and
    update then called y.combine_first(None).
    """
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.Series(np.arange(15, dtype=float))
    y_new = pd.DataFrame({"ilocs": [1]})

    detector = DummyRegularAnomalies()
    detector.fit(X.iloc[:10])  # fit without labels -> self._y = None
    detector.update(X.iloc[10:].to_frame(), y=y_new)  # was raising TypeError before fix
    assert detector._y is not None
