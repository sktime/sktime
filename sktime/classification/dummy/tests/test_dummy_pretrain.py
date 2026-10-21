"""Tests for DummyPretrainClassifier, and the pretrain logic of BaseClassifier."""

__author__ = ["srupat"]

import numpy as np
import pytest

from sktime.classification.dummy import DummyClassifier, DummyPretrainClassifier
from sktime.exceptions import NotFittedError
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(DummyPretrainClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _make_X(n_instances):
    """Make numpy3D X with n_instances univariate series of length 10."""
    rng = np.random.default_rng(42)
    return rng.random((n_instances, 1, 10))


# pretrain labels: "a" 6 times, "b" 2 times, "c" 4 times
Y_PRETRAIN = np.array(["a"] * 6 + ["b"] * 2 + ["c"] * 4)
X_PRETRAIN = _make_X(len(Y_PRETRAIN))
# fit labels: "a" 1 time, "b" 3 times
Y_FIT = np.array(["a"] + ["b"] * 3)
X_FIT = _make_X(len(Y_FIT))


def test_pretrain_counts_are_pseudo_counts():
    """Test predicted probabilities pool fit counts and weighted pretrain counts."""
    clf = DummyPretrainClassifier(pretrain_weight=0.5)
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)

    assert clf.n_pretrain_instances_ == 12
    assert clf.pretrain_class_counts_ == {"a": 6, "b": 2, "c": 4}

    clf.fit(X_FIT, Y_FIT)

    # "c" is not seen in fit, so it is not predicted
    # a: 1 + 0.5 * 6 = 4, b: 3 + 0.5 * 2 = 4
    np.testing.assert_array_equal(clf.classes_, ["a", "b"])
    np.testing.assert_array_almost_equal(clf.class_prior_, [0.5, 0.5])
    y_proba = clf.predict_proba(_make_X(3))
    np.testing.assert_array_almost_equal(y_proba, np.tile([0.5, 0.5], (3, 1)))


def test_no_pretrain_equals_prior_strategy():
    """Test that without pretraining, predictions equal the prior strategy."""
    clf = DummyPretrainClassifier().fit(X_FIT, Y_FIT)
    reference = DummyClassifier(strategy="prior").fit(X_FIT, Y_FIT)

    X_new = _make_X(3)
    np.testing.assert_array_almost_equal(
        clf.predict_proba(X_new), reference.predict_proba(X_new)
    )
    np.testing.assert_array_equal(clf.predict(X_new), reference.predict(X_new))


@pytest.mark.parametrize("pretrain_y", [False, True], ids=["y_none", "weight_0"])
def test_pretrain_without_effect(pretrain_y):
    """Test pretrain without labels, or with weight 0, does not change predictions."""
    if pretrain_y:
        clf = DummyPretrainClassifier(pretrain_weight=0)
        clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    else:
        clf = DummyPretrainClassifier()
        clf.pretrain(X_PRETRAIN)
        assert clf.pretrain_class_counts_ == {}
        assert clf.n_pretrain_instances_ == 12

    clf.fit(X_FIT, Y_FIT)
    np.testing.assert_array_almost_equal(clf.class_prior_, [0.25, 0.75])


def test_pretrain_update_accumulates():
    """Test that a second pretrain call adds to the counts of the first."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    clf.pretrain(X_FIT, Y_FIT)

    assert clf.state == "pretrained"
    assert clf.n_pretrain_instances_ == 16
    assert clf.pretrain_class_counts_ == {"a": 7, "b": 5, "c": 4}


def test_refit_retains_pretrained_state():
    """Test that fit, and refit, keep pretrained and refresh fitted attributes."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)

    clf.fit(X_FIT, Y_FIT)
    prior_first_fit = clf.class_prior_.copy()

    # refit on other labels: fitted state is refreshed, pretrained state kept
    y_other = np.array(["a", "c", "c", "c"])
    clf.fit(X_FIT, y_other)
    assert clf.pretrain_class_counts_ == {"a": 6, "b": 2, "c": 4}
    np.testing.assert_array_equal(clf.classes_, ["a", "c"])

    # refit on the first labels reproduces the first fit
    clf.fit(X_FIT, Y_FIT)
    np.testing.assert_array_almost_equal(clf.class_prior_, prior_first_fit)


def test_fitted_params_exclude_pretrained_params():
    """Test that fitted and pretrained parameters are inspected separately."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    clf.fit(X_FIT, Y_FIT)

    pretrained_params = clf.get_pretrained_params()
    fitted_params = clf.get_fitted_params()

    assert set(pretrained_params) == {"n_pretrain_instances_", "pretrain_class_counts_"}
    # keys of get_fitted_params are attribute names without the trailing "_"
    assert "class_prior" in fitted_params
    assert "n_pretrain_instances" not in fitted_params
    assert "pretrain_class_counts" not in fitted_params


def test_pretrain_after_fit_discards_fitted_state():
    """Test that pretrain after fit removes fitted attributes, and updates."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    clf.fit(X_FIT, Y_FIT)

    clf.pretrain(X_FIT, Y_FIT)

    assert clf.state == "pretrained"
    assert not clf.is_fitted
    assert not hasattr(clf, "class_prior_")
    # pretrained state was retained, so _pretrain_update added to it
    assert clf.pretrain_class_counts_ == {"a": 7, "b": 5, "c": 4}
    with pytest.raises(NotFittedError):
        clf.predict(X_FIT)


def test_pretrain_after_fit_without_pretraining():
    """Test that pretrain after fit, without prior pretrain, starts from scratch."""
    clf = DummyPretrainClassifier()
    clf.fit(X_FIT, Y_FIT)

    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)

    assert clf.state == "pretrained"
    assert not hasattr(clf, "class_prior_")
    assert clf.pretrain_class_counts_ == {"a": 6, "b": 2, "c": 4}


def test_clone_copies_pretrained_state():
    """Test that clone retains pretrained state, without shared references."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    clf.fit(X_FIT, Y_FIT)

    cloned = clf.clone()

    assert cloned.state == "pretrained"
    assert not hasattr(cloned, "class_prior_")
    assert cloned.pretrain_class_counts_ == clf.pretrain_class_counts_
    assert cloned.pretrain_class_counts_ is not clf.pretrain_class_counts_

    # pretraining the clone further does not affect the original
    cloned.pretrain(X_FIT, Y_FIT)
    assert clf.pretrain_class_counts_ == {"a": 6, "b": 2, "c": 4}


def test_reset_discards_pretrained_state():
    """Test that the public reset still returns to the post-init state."""
    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)
    clf.fit(X_FIT, Y_FIT)

    clf.reset()

    assert clf.state == "new"
    assert clf.get_pretrained_params() == {}
    assert not hasattr(clf, "pretrain_class_counts_")


@pytest.mark.parametrize("change_state", [True, False])
def test_fit_predict_retains_pretrained_state(change_state):
    """Test that fit_predict, also with cv, uses and retains pretrained state."""
    X = _make_X(8)
    y = np.array(["a", "b"] * 4)

    clf = DummyPretrainClassifier()
    clf.pretrain(X_PRETRAIN, Y_PRETRAIN)

    clf.fit_predict_proba(X, y, cv=2, change_state=change_state)

    assert clf.pretrain_class_counts_ == {"a": 6, "b": 2, "c": 4}
    if change_state:
        assert clf.state == "fitted"
        # a: 4 + 6 = 10, b: 4 + 2 = 6
        np.testing.assert_array_almost_equal(clf.class_prior_, [10 / 16, 6 / 16])
    else:
        assert clf.state == "pretrained"


def test_pretrain_rejects_vectorization():
    """Test that pretrain raises on y with several columns, without multioutput."""
    clf = DummyPretrainClassifier()
    y_2d = np.column_stack([Y_PRETRAIN, Y_PRETRAIN])

    with pytest.raises(TypeError, match="does not support automatic vectorization"):
        clf.pretrain(X_PRETRAIN, y_2d)

    assert clf.state == "new"
