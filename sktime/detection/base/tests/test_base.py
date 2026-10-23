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


def test_remember_data_config():
    """Test remember_data configuration on BaseDetector."""
    from sktime.detection.dummy import DummyRegularAnomalies

    model = DummyRegularAnomalies()
    assert model.get_config()["remember_data"] is True

    model = DummyRegularAnomalies().set_config(remember_data=False)
    assert model.get_config()["remember_data"] is False


def test_remember_data_storage():
    """Test data storage during fit and update."""
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.DataFrame({"value": range(5)}, index=range(5))
    X_new = pd.DataFrame({"value": range(5, 8)}, index=range(5, 8))

    model = DummyRegularAnomalies().set_config(remember_data=True)
    model.fit(X)
    assert len(model._X) == len(X)

    model.update(X_new)
    assert len(model._X) == len(X) + len(X_new)

    model = DummyRegularAnomalies().set_config(remember_data=False)
    model.fit(X)
    assert model._X is None


def test_update_without_remember_data_warns():
    """Test fallback update warns when data is not remembered."""
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.DataFrame({"value": range(5)})
    X_new = pd.DataFrame({"value": range(5, 8)})

    model = DummyRegularAnomalies().set_config(remember_data=False)
    model.fit(X)

    with pytest.warns(UserWarning, match="remember_data=False"):
        model.update(X_new)
