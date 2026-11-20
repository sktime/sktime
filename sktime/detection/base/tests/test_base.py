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
@pytest.mark.parametrize("remember_data", [True, False])
def test_remember_data(remember_data):
    """Test that the ``remember_data`` config works as expected.

    True: data is stored as self._X, self._y.
    False: data is not stored, and update warns instead of refitting.
    """
    from sktime.detection.dummy import DummyRegularAnomalies
    from sktime.utils._testing.detection import make_detection_problem

    X = make_detection_problem(n_timepoints=20, random_state=0).to_frame()
    X_fit, X_new = X.iloc[:15], X.iloc[15:]
    y = pd.DataFrame({"ilocs": [5]})

    model = DummyRegularAnomalies().set_config(remember_data=remember_data)
    model.fit(X_fit, y)

    if remember_data:
        assert model._X is X_fit
        assert model._y is y
    else:
        assert not hasattr(model, "_X")
        assert not hasattr(model, "_y")

        # update should not fail, but warn that the model is left unchanged
        with pytest.warns(UserWarning, match="remember_data=False"):
            model.update(X_new)

        assert not hasattr(model, "_X")
        assert not hasattr(model, "_y")
