# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly"]
__all__ = []

import pytest
from sklearn.preprocessing import StandardScaler

from sktime.clustering.compose import ClustererPipeline
from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sktime.dists_kernels import FlatDist
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils._testing.panel import _make_panel_X


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.clustering", "sktime.distances"]),
    reason="run test only if clustering or distances code has changed",
)
def test_dunder_mul():
    """Test the mul dunder method."""
    RAND_SEED = 42
    X = _make_panel_X(n_instances=10, n_timepoints=12, random_state=RAND_SEED)
    X_test = X

    t1 = ExponentTransformer(power=4)
    t2 = ExponentTransformer(power=0.25)

    c = TimeSeriesDBSCAN(FlatDist.create_test_instance(), eps=4, min_samples=1)
    t12c_1 = t1 * (t2 * c)
    t12c_2 = (t1 * t2) * c
    t12c_3 = t1 * t2 * c

    assert isinstance(t12c_1, ClustererPipeline)
    assert isinstance(t12c_2, ClustererPipeline)
    assert isinstance(t12c_3, ClustererPipeline)

    y_pred = c.fit(X).predict(X_test)

    _assert_array_almost_equal(y_pred, t12c_1.fit(X).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_2.fit(X).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_3.fit(X).predict(X_test))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.clustering", "sktime.distances"]),
    reason="run test only if clustering or distances code has changed",
)
def test_mul_sklearn_autoadapt():
    """Test auto-adapter for sklearn in mul."""
    RAND_SEED = 42
    X = _make_panel_X(n_instances=10, n_timepoints=12, random_state=RAND_SEED)
    X_test = X

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    c = TimeSeriesDBSCAN(FlatDist.create_test_instance(), eps=4, min_samples=1)

    t12c_1 = t1 * (t2 * c)
    t12c_2 = (t1 * t2) * c
    t12c_3 = t1 * t2 * c

    assert isinstance(t12c_1, ClustererPipeline)
    assert isinstance(t12c_2, ClustererPipeline)
    assert isinstance(t12c_3, ClustererPipeline)

    y_pred = t12c_1.fit(X).predict(X_test)

    _assert_array_almost_equal(y_pred, t12c_2.fit(X).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_3.fit(X).predict(X_test))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.clustering", "sktime.distances"]),
    reason="run test only if clustering or distances code has changed",
)
def test_missing_unequal_tag_inference():
    """Test that ClustererPipeline infers missing/unequal tags correctly."""
    c = TimeSeriesDBSCAN(FlatDist.create_test_instance())
    c1 = ExponentTransformer() * PaddingTransformer() * ExponentTransformer() * c
    c2 = ExponentTransformer() * ExponentTransformer() * c
    c3 = Imputer() * ExponentTransformer() * c
    c4 = ExponentTransformer() * Imputer() * c

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")
    assert c3.get_tag("capability:missing_values")
    assert not c4.get_tag("capability:missing_values")
