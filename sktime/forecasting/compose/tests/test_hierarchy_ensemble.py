#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of HierarchyEnsembleForecaster functionality."""

__author__ = ["VyomkeshVyas"]

import numpy as np
import pytest

from sktime.base._meta import flatten
from sktime.datatypes._utilities import get_window
from sktime.forecasting.compose import HierarchyEnsembleForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen, _make_hierarchical


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [("ptf", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
        [("naive", NaiveForecaster(), 0)],
    ],
)
def test_hierarchy_ensemble_level_predict(forecasters):
    """Check the level predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters, default=forecasters[0][1].clone()
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[i][1])]
        test_frcstr.fit(df, fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Level predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[-1][1])]
    def_frcstr.fit(df, fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Level default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [("ptf", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
        [("naive", NaiveForecaster(), 0)],
    ],
)
def test_hierarchy_ensemble_level_predict_parallel(forecasters):
    """Check the level predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters,
        default=forecasters[0][1].clone(),
        by="level",
        backend="joblib",
        backend_params={"backend": "loky", "n_jobs": -1},
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[i][1])]
        test_frcstr.fit(df, fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Level predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[-1][1])]
    def_frcstr.fit(df, fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Level default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("ptf", PolynomialTrendForecaster(), [("__total", "__total")]),
            ("naive", NaiveForecaster(), [("l2_node01", "__total")]),
        ],
        [("naive", NaiveForecaster(), [("l2_node01", "l1_node06")])],
    ],
)
def test_hierarchy_ensemble_node_predict_parallel(forecasters):
    """Check the node predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters,
        by="node",
        default=forecasters[0][1].clone(),
        backend="joblib",
        backend_params={"backend": "loky", "n_jobs": -1},
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[i][1])]
        test_frcstr.fit(df, fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Node predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[-1][1])]
    def_frcstr.fit(df, fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Node default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("ptf", PolynomialTrendForecaster(), [("__total", "__total")]),
            ("naive", NaiveForecaster(), [("l2_node01", "__total")]),
        ],
        [("naive", NaiveForecaster(), [("l2_node01", "l1_node06")])],
    ],
)
def test_hierarchy_ensemble_node_predict(forecasters):
    """Check the node predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters, by="node", default=forecasters[0][1].clone()
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[i][1])]
        test_frcstr.fit(df, fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Node predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[-1][1])]
    def_frcstr.fit(df, fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Node default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [("ptf", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
        [("naive", NaiveForecaster(), 0)],
    ],
)
def test_hierarchy_ensemble_exog(forecasters):
    """Check exog variable functionality."""
    y = _make_hierarchical(
        hierarchy_levels=(2, 4),
        n_columns=2,
        min_timepoints=24,
        max_timepoints=24,
        index_type="period",
    )

    y_train = get_window(y, lag=2)
    y_test = get_window(y, window_length=2)

    X = _make_hierarchical(
        hierarchy_levels=(2, 4),
        n_columns=2,
        min_timepoints=24,
        max_timepoints=24,
        index_type="period",
    )
    X.columns = ["foo", "bar"]

    X_train = get_window(X, lag=2)
    X_test = get_window(X, window_length=2)

    estimator_instance = HierarchyEnsembleForecaster(
        forecasters=forecasters, default=NaiveForecaster()
    )
    estimator_instance.fit(y=y_train, X=X_train, fh=[1, 2, 3])
    estimator_instance.predict(X=X_test)
    estimator_instance.update(y=y_test, X=X_test)


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("trend", TrendForecaster(), ["l1_node01"]),
            ("polytrend", PolynomialTrendForecaster(), ["l1_node02", "l1_node03"]),
            ("naive", NaiveForecaster(), ["__total"]),
        ],
        [
            (
                "trend",
                TrendForecaster(),
                [("__total"), ("l1_node01"), ("l1_node02"), ("l1_node03")],
            ),
        ],
    ],
)
@pytest.mark.parametrize("default", [NaiveForecaster(), None])
def test_level_one_data(forecasters, default):
    """Check for data with one level of hierarchy (excluding timepoints level)."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(forecasters, by="node", default=default)

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[i][1])]
        test_frcstr.fit(df, fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Node predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    _, _, nodes = zip(*forecasters)
    nodes = set(flatten(nodes))
    if default is not None and len(nodes) != len(y.index.droplevel(-1).unique()):
        def_frcstr = default
        df = y[y.index.droplevel(-1).isin(forecaster.fitted_list[-1][1])]
        def_frcstr.fit(df, fh=[1, 2, 3])
        def_pred = def_frcstr.predict()
        msg = "Node default predictions do not match"
        assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.skipif(
    not run_test_for_class(HierarchyEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_get_fitted_params():
    """Tests that get_fitted_params works as expected.

    Checks absence of bug #7418.
    """
    from sktime.forecasting.compose import HierarchyEnsembleForecaster
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
    from sktime.utils._testing.hierarchical import _bottom_hier_datagen

    y = _bottom_hier_datagen(no_bottom_nodes=7, no_levels=2, random_seed=123)

    forecasters = [
        ("naive", NaiveForecaster(), 0),
        ("trend", TrendForecaster(), 1),
    ]
    forecaster = HierarchyEnsembleForecaster(
        forecasters=forecasters,
        by="level",
        default=PolynomialTrendForecaster(degree=2),
    )
    forecaster.fit(y, fh=[1, 2, 3])

    assert forecaster.get_fitted_params()["naive"].is_fitted
    assert forecaster.get_fitted_params()["trend"].is_fitted
