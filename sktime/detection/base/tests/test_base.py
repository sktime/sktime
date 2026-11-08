"""Tests for base change point detection class."""

import numpy as np
import pytest

from sktime.detection.dummy import DummyRegularAnomalies
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


@pytest.mark.parametrize("ndim", [1, 2])
def test_repeated_numpy_updates_after_indexed_fit(ndim):
    """Successive array batches append after the stored index without mutation."""
    import pandas as pd

    initial = pd.DataFrame(np.arange(10.0), index=pd.RangeIndex(100, 110))
    detector = DummyRegularAnomalies().fit(initial)
    for batch in [np.arange(10.0, 13.0), np.arange(13.0, 15.0)]:
        if ndim == 2:
            batch = batch.reshape(-1, 1)
        original = batch.copy()
        detector.update(batch)
        np.testing.assert_array_equal(batch, original)
    pd.testing.assert_index_equal(detector._X.index, pd.RangeIndex(100, 115))
    np.testing.assert_array_equal(detector._X.to_numpy().ravel(), np.arange(15.0))


def test_indexed_update_retains_overlap_semantics():
    """Explicit pandas labels revise existing observations and append new ones."""
    import pandas as pd

    initial = pd.DataFrame({0: [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3))
    revised = pd.DataFrame({0: [30.0, 4.0]}, index=pd.RangeIndex(2, 4))
    original = revised.copy(deep=True)
    detector = DummyRegularAnomalies().fit(initial).update(revised)
    expected = pd.DataFrame({0: [1.0, 2.0, 30.0, 4.0]})
    pd.testing.assert_frame_equal(detector._X, expected)
    pd.testing.assert_frame_equal(revised, original)
