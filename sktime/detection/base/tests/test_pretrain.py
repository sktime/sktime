"""Tests for pretrain on BaseDetector, for detectors without pretrain capability."""

__author__ = ["yash-sangwan"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import (
    DummyRegularAnomalies,
    ZeroChangePoints,
    ZeroSegments,
)
from sktime.exceptions import NotFittedError
from sktime.forecasting.base._clone_plugin import _PretrainedCloner
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.detection import make_detection_problem
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.deep_equals import deep_equals

# one detector per task, with and without fit_is_empty
DETECTORS = [DummyRegularAnomalies, ZeroChangePoints, ZeroSegments]

pytestmark = pytest.mark.skipif(
    not run_test_module_changed(["sktime.detection", "sktime.forecasting.base"]),
    reason="module not changed",
)


def _make_panel(random_state=0):
    """Make a panel of 3 time series with 20 time points each."""
    return _make_hierarchical(
        hierarchy_levels=(3,),
        min_timepoints=20,
        max_timepoints=20,
        random_state=random_state,
    )


def _assert_only_state_changed(detector, attrs_before):
    """Assert pretrain stored nothing: same attributes, only _state may differ."""
    attrs_after = vars(detector)
    assert attrs_after.keys() == attrs_before.keys()
    for key, value in attrs_before.items():
        if key != "_state":
            assert attrs_after[key] is value, key


@pytest.mark.parametrize("detector_cls", DETECTORS)
@pytest.mark.parametrize(
    "X_panel",
    [_make_panel(), np.random.default_rng(0).random((3, 1, 20))],
    ids=["pd-multiindex", "numpy3D"],
)
@pytest.mark.parametrize(
    "y", [None, pd.DataFrame({"ilocs": [3, 7]})], ids=["y_none", "y_events"]
)
def test_pretrain_is_noop(detector_cls, X_panel, y):
    """Test pretrain is callable, and a no-op apart from the state."""
    detector = detector_cls.create_test_instance()
    assert detector.get_tag("capability:pretrain") is False
    assert detector.state == "new"
    attrs_before = dict(vars(detector))

    result = detector.pretrain(X_panel, y=y)

    assert result is detector
    assert detector.state == "pretrained"
    assert not detector.is_fitted
    assert detector.get_pretrained_params() == {}
    _assert_only_state_changed(detector, attrs_before)


@pytest.mark.parametrize("detector_cls", DETECTORS)
def test_pretrain_state(detector_cls):
    """Test state moves new, pretrained, fitted, and back to pretrained."""
    detector = detector_cls.create_test_instance()
    X_train = make_detection_problem(n_timepoints=30, random_state=0)
    X_test = make_detection_problem(n_timepoints=10, random_state=1)

    assert detector.state == "new"

    detector.pretrain(_make_panel())
    assert detector.state == "pretrained"
    assert not detector.is_fitted
    with pytest.raises(NotFittedError):
        detector.predict(X_test)

    detector.fit(X_train)
    assert detector.state == "fitted"
    assert detector.is_fitted

    # as for forecasters, pretrain after fit sets the state to pretrained
    detector.pretrain(_make_panel())
    assert detector.state == "pretrained"
    assert not detector.is_fitted


@pytest.mark.parametrize("detector_cls", DETECTORS)
def test_pretrain_twice(detector_cls):
    """Test a second pretrain call is also a no-op."""
    detector = detector_cls.create_test_instance()
    detector.pretrain(_make_panel(random_state=0))
    attrs_before = dict(vars(detector))

    detector.pretrain(_make_panel(random_state=1))

    assert detector.state == "pretrained"
    assert detector.get_pretrained_params() == {}
    _assert_only_state_changed(detector, attrs_before)


@pytest.mark.parametrize("detector_cls", DETECTORS)
def test_clone_after_pretrain(detector_cls):
    """Test clone uses the pretrain clone plugin, and returns a new detector."""
    detector = detector_cls.create_test_instance()
    assert _PretrainedCloner in detector._get_clone_plugins()

    detector.pretrain(_make_panel())
    detector_clone = detector.clone()

    assert detector_clone is not detector
    assert detector_clone.state == "new"
    assert detector_clone.get_pretrained_params() == {}
    assert detector_clone.get_params() == detector.get_params()
    assert detector.state == "pretrained"

    detector.fit(make_detection_problem(n_timepoints=30, random_state=0))
    assert detector.clone().state == "new"


@pytest.mark.parametrize("detector_cls", DETECTORS)
def test_pretrain_then_fit_predict(detector_cls):
    """Test fit and predict after pretrain give the same result as without it."""
    X_train = make_detection_problem(n_timepoints=30, random_state=0)
    X_test = make_detection_problem(n_timepoints=10, random_state=1)

    detector = detector_cls.create_test_instance()
    detector.pretrain(_make_panel())
    detector.fit(X_train)
    y_pred = detector.predict(X_test)

    reference = detector_cls.create_test_instance()
    reference.fit(X_train)
    y_pred_reference = reference.predict(X_test)

    assert detector.state == "fitted"
    pd.testing.assert_frame_equal(y_pred, y_pred_reference)


@pytest.mark.parametrize("detector_cls", DETECTORS)
@pytest.mark.parametrize(
    "X_single",
    [
        make_detection_problem(n_timepoints=20, random_state=0),
        make_detection_problem(n_timepoints=20, random_state=0).to_frame(),
        np.arange(20.0),
    ],
    ids=["pd.Series", "pd.DataFrame", "np.ndarray"],
)
def test_pretrain_single_series_raises(detector_cls, X_single):
    """Test pretrain raises TypeError on a single series, and keeps the state."""
    detector = detector_cls.create_test_instance()

    with pytest.raises(TypeError, match="single Series"):
        detector.pretrain(X_single)

    assert detector.state == "new"


@pytest.mark.parametrize("detector_cls", DETECTORS)
def test_pretrain_hierarchical_raises(detector_cls):
    """Test pretrain raises TypeError on hierarchical data, and keeps the state."""
    detector = detector_cls.create_test_instance()
    X_hier = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=10, max_timepoints=10
    )

    with pytest.raises(TypeError, match="Panel data only, not Hierarchical"):
        detector.pretrain(X_hier)

    assert detector.state == "new"


def _make_fit_and_new_data():
    """Make DataFrame data for fit, and new data that continues its index.

    Uses DataFrame input, as update with Series input fails in combine_first,
    for a reason unrelated to pretrain.
    """
    X_all = make_detection_problem(n_timepoints=40, random_state=0).to_frame()
    return X_all.iloc[:30], X_all.iloc[30:]


def test_pretrain_then_fit_update():
    """Test update after pretrain and fit gives the same result as without it."""
    X_fit, X_new = _make_fit_and_new_data()

    detector = DummyRegularAnomalies()
    detector.pretrain(_make_panel())
    detector.fit(X_fit)
    result = detector.update(X_new)

    reference = DummyRegularAnomalies()
    reference.fit(X_fit)
    reference.update(X_new)

    assert result is detector
    assert detector.state == "fitted"
    is_equal, msg = deep_equals(vars(detector), vars(reference), return_msg=True)
    assert is_equal, msg
    pd.testing.assert_frame_equal(detector.predict(X_new), reference.predict(X_new))


def test_pretrain_then_fit_update_predict():
    """Test update_predict after pretrain and fit gives the same result as without."""
    X_fit, X_new = _make_fit_and_new_data()

    detector = DummyRegularAnomalies()
    detector.pretrain(_make_panel())
    detector.fit(X_fit)
    y_pred = detector.update_predict(X_new)

    reference = DummyRegularAnomalies()
    reference.fit(X_fit)
    y_pred_reference = reference.update_predict(X_new)

    assert detector.state == "fitted"
    pd.testing.assert_frame_equal(y_pred, y_pred_reference)
    is_equal, msg = deep_equals(vars(detector), vars(reference), return_msg=True)
    assert is_equal, msg
