#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

import numpy as np
import pytest

from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import TransformByLevel
from sktime.transformations.impute import Imputer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.hierarchical import _make_hierarchical

y, X = make_forecasting_problem(make_X=True)

X.iloc[3, 0] = np.nan
X.iloc[3, 1] = np.nan
X.iloc[0, 1] = np.nan
X.iloc[-1, 1] = np.nan

y.iloc[3] = np.nan
y.iloc[0] = np.nan
y.iloc[-1] = np.nan

z = _make_hierarchical(hierarchy_levels=(3,), n_columns=3)

z.iloc[3] = np.nan
z.iloc[0] = np.nan
z.iloc[-1] = np.nan


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("forecaster", [None, "NaiveForecaster"])
@pytest.mark.parametrize("value", [None, 1])
@pytest.mark.parametrize("Z", [y, X, z])
@pytest.mark.parametrize(
    "method",
    [
        "drift",
        "linear",
        "nearest",
        "constant",
        "mean",
        "median",
        "backfill",
        "pad",
        "random",
        "forecaster",
    ],
)
def test_imputer(method, Z, value, forecaster):
    """Test univariate and multivariate Imputer with all methods."""
    if forecaster == "NaiveForecaster" or method == "forecaster":
        from sktime.forecasting.naive import NaiveForecaster

        forecaster = NaiveForecaster()

    value = 1 if method == "constant" else value
    t = Imputer(method=method, forecaster=forecaster, value=value)
    y_hat = t.fit_transform(Z)
    assert not y_hat.isnull().to_numpy().any()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "method",
    [
        "linear",
        "nearest",
        "mean",
        "median",
        "backfill",
        "pad",
    ],
)
def test_impute_multiindex(method):
    """Test for data leakage in case of pd-multiindex data.

    Failure case in bug #6224
    """
    df = get_examples(mtype="pd-multiindex")[0].copy()
    df.iloc[:3, :] = np.nan  # instance 0 entirely missing
    df.iloc[3:4, :] = np.nan  # instance 1 first timepoint missing
    df.iloc[8:, :] = np.nan  # instance 2 last timepoint missing

    imp = Imputer(method=method)
    df_imp = imp.fit_transform(df)

    # instance 0 entirely missing, so it should remain missing
    assert np.array_equal(df.iloc[:3, :], df_imp.iloc[:3, :], equal_nan=True)

    # instance 1 and 2 should not have any missing values
    assert not df_imp.iloc[3:, :].isna().any().any()

    # test consistency between applying the imputer to every instance separately,
    # vs applying them to the panel
    imp_tbl = TransformByLevel(Imputer(method=method))
    df_imp_tbl = imp_tbl.fit_transform(df)
    assert np.array_equal(df_imp, df_imp_tbl, equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_forecaster_y():
    """Test that forecaster imputer works with y.

    Failure case in bug #5284.
    """
    from sklearn.linear_model import LinearRegression

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import YfromX

    X = load_airline()
    y = load_airline()

    model_reg = YfromX(LinearRegression())
    model_reg.fit(X, y)
    transformer = Imputer(method="forecaster", forecaster=model_reg)

    transformer.fit(X=X, y=y)
    transformer.transform(X=X, y=y)


def _panel_frame(values, instance_ids, columns=("a", "b")):
    """Build a ``pd-multiindex`` panel from ``(instance, time, variable)`` values."""
    import pandas as pd

    n_instances, n_time, n_vars = values.shape
    if len(instance_ids) != n_instances:
        raise ValueError("instance_ids must match the first axis of values")
    index = pd.MultiIndex.from_product(
        [list(instance_ids), range(n_time)], names=["instance", "time"]
    )
    flat = values.reshape(n_instances * n_time, n_vars)
    return pd.DataFrame(flat, index=index, columns=list(columns))


def _per_instance_transform(method, train, test, **kwargs):
    """Fit and transform each instance on its own, in instance order."""
    import pandas as pd

    pieces = []
    ids = test.index.droplevel(-1).unique()
    train_ids = train.index.droplevel(-1).unique()
    for inst in ids:
        test_i = test.xs(inst, drop_level=True)
        imp = Imputer(method=method, **kwargs)
        if inst in train_ids:
            train_i = train.xs(inst, drop_level=True)
            out_i = imp.fit(train_i).transform(test_i)
        else:
            out_i = imp.fit_transform(test_i)
        out_i = out_i.copy()
        out_i.index = test.xs(inst, drop_level=False).index
        pieces.append(out_i)
    return pd.concat(pieces)


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "method",
    ["drift", "linear", "nearest", "mean", "median", "random", "ffill", "constant"],
)
def test_imputer_transform_different_n_instances(method):
    """Imputer.transform must accept a panel larger than the one seen in fit.

    Failure mode of bug #3588: vectorized methods raised ``RuntimeError`` when
    the number of instances changed. ``mean`` and ``median`` did not raise, but
    filled unseen instances by forward/back fill instead of the fitted statistic.
    """
    rng = np.random.default_rng(0)
    train = rng.random((3, 2, 8))
    test = rng.random((5, 2, 8))
    # NaNs on an instance index that also occurred in fit, and on a new one.
    test[1, :, 3] = np.nan
    test[4, 0, [0, 2, 5]] = np.nan
    test[4, 1, 1] = np.nan

    kwargs = {}
    if method == "constant":
        kwargs["value"] = -1.0
    if method == "random":
        kwargs["random_state"] = 0

    out = Imputer(method=method, **kwargs).fit(train).transform(test)

    assert out.shape == test.shape
    # Constant fill and in-sample interpolation can always remove these NaNs.
    # Entirely missing series are not constructed here.
    assert not np.isnan(out).any()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_issue_3588_numpy_panel():
    """Reproduce bug #3588: default Imputer on a wider numpy panel."""
    train = np.random.default_rng(1).random((3, 10, 20))
    test = np.random.default_rng(2).random((5, 10, 20))
    test[2, :, 5] = np.nan

    out = Imputer().fit(train).transform(test)

    assert out.shape == (5, 10, 20)
    assert not np.isnan(out).any()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["drift", "linear", "nearest", "mean", "random"])
def test_imputer_same_n_instances_matches_per_instance(method):
    """Same-sized panels keep the historical per-instance imputation."""
    rng = np.random.default_rng(3)
    train_vals = rng.random((3, 2, 7))
    test_vals = rng.random((3, 2, 7))
    test_vals[0, 0, 2] = np.nan
    test_vals[1, :, 0] = np.nan
    test_vals[2, 1, -1] = np.nan
    ids = ["a", "b", "c"]
    train = _panel_frame(train_vals.transpose(0, 2, 1), ids)
    test = _panel_frame(test_vals.transpose(0, 2, 1), ids)
    kwargs = {"random_state": 0} if method == "random" else {}

    out = Imputer(method=method, **kwargs).fit(train).transform(test)
    expected = _per_instance_transform(method, train, test, **kwargs)

    assert out.index.equals(test.index)
    assert np.allclose(out.to_numpy(), expected.to_numpy(), equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["mean", "median"])
def test_imputer_mean_median_unseen_instances_use_pooled_fit_statistic(method):
    """Unseen instances use the pooled fit statistic, not neighbour fill.

    An instance that was entirely missing at ``fit`` stays missing, matching
    the per-instance behaviour guarded by bug #6224.
    """
    import pandas as pd

    index = pd.MultiIndex.from_product(
        [["a", "b", "allnan"], range(4)], names=["instance", "time"]
    )
    train = pd.DataFrame(
        {
            "x": [
                0.0,
                2.0,
                4.0,
                np.nan,
                10.0,
                10.0,
                10.0,
                10.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=index,
    )
    # a mean = 2, b mean = 10, pooled = (0+2+4+10+10+10+10) / 7 = 46/7
    test_index = pd.MultiIndex.from_product(
        [["b", "c"], range(3)], names=["instance", "time"]
    )
    test = pd.DataFrame({"x": [np.nan, 8.0, 8.0, np.nan, 1.0, 1.0]}, index=test_index)

    out = Imputer(method=method).fit(train).transform(test)

    stat = getattr(train["x"], method)()
    # Seen instance "b" keeps its own fitted statistic.
    assert out.loc[("b", 0), "x"] == getattr(train.xs("b")["x"], method)()
    assert out.loc[("b", 1), "x"] == 8.0
    # Unseen instance "c" uses the pooled statistic, not the neighbouring 1.0.
    assert out.loc[("c", 0), "x"] == stat
    assert out.loc[("c", 1), "x"] == 1.0

    # The all-missing training instance is unchanged by fit_transform.
    refit = Imputer(method=method).fit_transform(train)
    assert np.isnan(refit.xs("allnan").to_numpy()).all()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["linear", "nearest", "drift"])
def test_imputer_unseen_instance_matches_self_imputation(method):
    """An instance id absent from fit is imputed from that series alone."""
    train_vals = np.array(
        [
            [[0.0, 1.0, 2.0, 3.0, 4.0]],
            [[2.0, 2.0, 2.0, 2.0, 2.0]],
        ]
    )
    # One more instance than fit, so this is not the equal-count positional map.
    test_vals = np.array(
        [
            [[0.0, np.nan, 2.0, np.nan, 4.0]],
            [[10.0, np.nan, 30.0, np.nan, 50.0]],
            [[4.0, np.nan, 0.0, np.nan, -4.0]],
        ]
    )
    # values layout above is (instance, variable, time); _panel_frame wants
    # (instance, time, variable)
    train = _panel_frame(train_vals.transpose(0, 2, 1), ["a", "b"], columns=["x"])
    test = _panel_frame(
        test_vals.transpose(0, 2, 1), ["a", "new", "extra"], columns=["x"]
    )

    out = Imputer(method=method).fit(train).transform(test)

    seen = Imputer(method=method).fit(train.xs("a", drop_level=True)).transform(
        test.xs("a", drop_level=True)
    )
    unseen = Imputer(method=method).fit_transform(test.xs("new", drop_level=True))

    assert np.allclose(out.xs("a").to_numpy(), seen.to_numpy(), equal_nan=True)
    assert np.allclose(out.xs("new").to_numpy(), unseen.to_numpy(), equal_nan=True)
    # Interior linear gap between 0 and 2 is 1, not a training-series value.
    if method == "linear":
        assert out.loc[("new", 1), "x"] == 20.0


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_forecaster_different_n_instances():
    """Forecaster imputation follows the same seen/unseen instance rules."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    forecaster = PolynomialTrendForecaster(degree=1)
    train_vals = np.arange(12, dtype=float).reshape(2, 6, 1)
    test_vals = np.arange(18, dtype=float).reshape(3, 6, 1)
    test_vals[0, 2, 0] = np.nan
    test_vals[2, 3, 0] = np.nan
    train = _panel_frame(train_vals, ["a", "b"], columns=["x"])
    test = _panel_frame(test_vals, ["a", "b", "c"], columns=["x"])

    out = Imputer(method="forecaster", forecaster=forecaster).fit(train).transform(test)

    assert not out.isna().any().any()
    seen = (
        Imputer(method="forecaster", forecaster=forecaster.clone())
        .fit(train.xs("a", drop_level=True))
        .transform(test.xs("a", drop_level=True))
    )
    unseen = Imputer(method="forecaster", forecaster=forecaster.clone()).fit_transform(
        test.xs("c", drop_level=True)
    )
    assert np.allclose(out.xs("a").to_numpy(), seen.to_numpy())
    assert np.allclose(out.xs("c").to_numpy(), unseen.to_numpy())


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_series_fit_broadcasts_to_panel():
    """A model fitted on one series can transform a panel of several series."""
    import pandas as pd

    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    index = pd.MultiIndex.from_product(
        [["a", "b"], range(5)], names=["instance", "time"]
    )
    test = pd.DataFrame(
        {"x": [0.0, np.nan, 2.0, 3.0, 4.0, 0.0, 10.0, np.nan, 30.0, 40.0]},
        index=index,
    )

    out = Imputer(method="linear").fit(train).transform(test)

    assert out.loc[("a", 1), "x"] == 1.0
    assert out.loc[("b", 2), "x"] == 20.0
    assert not out.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_panel_fit_then_series_transform():
    """A series has no instance id, so panel statistics are pooled or refit."""
    import pandas as pd

    index = pd.MultiIndex.from_product(
        [["a", "b"], range(4)], names=["instance", "time"]
    )
    train = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0]}, index=index)
    series = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})

    mean_out = Imputer(method="mean").fit(train).transform(series)
    # Pooled mean of the panel is 4, not a neighbour fill of 1 or 3.
    assert mean_out.loc[1, "x"] == 4.0
    assert mean_out.loc[3, "x"] == 4.0

    drift_out = Imputer(method="drift").fit(train).transform(series)
    self_fit = Imputer(method="drift").fit_transform(series)
    assert np.allclose(drift_out.to_numpy(), self_fit.to_numpy())


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_numpy_series_fit_transforms_numpy_panel():
    """Column names differ between a numpy series (``0``) and panel (``var_0``)."""
    series = np.arange(6, dtype=float)
    panel = np.arange(18, dtype=float).reshape(3, 1, 6)
    panel[1, 0, 2] = np.nan
    panel[2, 0, 0] = np.nan

    drift = Imputer(method="drift").fit(series).transform(panel)
    assert drift.shape == panel.shape
    assert not np.isnan(drift).any()
    # Training series is 0,1,2,3,4,5, so the missing interior point is 2.
    assert drift[1, 0, 2] == 2.0

    mean = Imputer(method="mean").fit(series).transform(panel)
    assert mean[1, 0, 2] == series.mean()
    assert not np.isnan(mean[2]).any()
