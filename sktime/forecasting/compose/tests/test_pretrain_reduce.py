"""Tests for pretrain-aware reduction forecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.base import BaseObject
from sklearn.base import BaseEstimator, RegressorMixin

from sktime.forecasting.compose import (
    BaseWindowNormalizer,
    MeanWindowNormalizer,
    MinMaxWindowNormalizer,
    ReductionForecaster,
    SubtractMeanNormalizer,
    ZScoreWindowNormalizer,
)
from sktime.forecasting.compose._pretrain_reduce import (
    _build_supervised_table,
    _resolve_normalizer,
)
from sktime.registry import scitype


class RecordingRegressor(BaseEstimator, RegressorMixin):
    """Regressor that records fit/predict calls and predicts a constant."""

    total_fit_calls = 0
    predict_log = []

    def __init__(self, constant=0.0, label="reg"):
        self.constant = constant
        self.label = label

    def fit(self, X, y):
        type(self).total_fit_calls += 1
        self.X_ = np.asarray(X, dtype=float).copy()
        self.y_ = np.asarray(y, dtype=float).copy()
        self.n_features_in_ = X.shape[1]
        self.fit_shape_ = X.shape
        self.target_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        type(self).predict_log.append((self.label, self.constant, X.copy()))
        return np.repeat(float(self.constant), X.shape[0])


class LoggingNormalizer(BaseWindowNormalizer):
    """Normalizer that records row-wise transform call order."""

    calls = []

    def _transform(self, lags, target=None):
        """Transform one row and record the private hook call."""
        type(self).calls.append((tuple(np.asarray(lags, dtype=float)), float(target)))
        return super()._transform(lags, target)


class BatchHookNormalizer(BaseWindowNormalizer):
    """Normalizer that records batch hook usage and rejects scalar hooks."""

    batch_transform_targets = []
    batch_inverse_calls = 0
    scalar_transform_calls = 0
    scalar_inverse_calls = 0

    @classmethod
    def reset(cls):
        """Reset class-level call logs."""
        cls.batch_transform_targets = []
        cls.batch_inverse_calls = 0
        cls.scalar_transform_calls = 0
        cls.scalar_inverse_calls = 0

    def _transform(self, lags, target=None):
        """Reject scalar private transform calls."""
        type(self).scalar_transform_calls += 1
        raise AssertionError("_transform should not be used")

    def _inverse_transform(self, y, lags):
        """Reject scalar private inverse calls."""
        type(self).scalar_inverse_calls += 1
        raise AssertionError("_inverse_transform should not be used")

    def _batch_transform(self, lags, target=None):
        """Transform in batch and record target blocks."""
        if target is None:
            type(self).batch_transform_targets.append(None)
            return lags + 10.0, None

        type(self).batch_transform_targets.append(target.copy())
        return lags + 10.0, target + 100.0

    def _batch_inverse_transform(self, y, lags):
        """Invert in batch and record calls."""
        type(self).batch_inverse_calls += 1
        return y + 5.0


def _panel_series():
    idx = pd.MultiIndex.from_product(
        [["a", "b"], pd.RangeIndex(8)], names=["id", "time"]
    )
    values = np.r_[np.arange(8, dtype=float), np.arange(10, 18, dtype=float)]
    return pd.DataFrame({"y": values}, index=idx)


def _panel_X():
    idx = pd.MultiIndex.from_product(
        [["a", "b"], pd.RangeIndex(8)], names=["id", "time"]
    )
    values = np.r_[np.arange(100, 108, dtype=float), np.arange(200, 208, dtype=float)]
    return pd.DataFrame({"x": values}, index=idx)


def _small_panel_series():
    idx = pd.MultiIndex.from_product(
        [["a", "b"], pd.RangeIndex(6)], names=["id", "time"]
    )
    values = np.r_[np.arange(6, dtype=float), np.arange(10, 16, dtype=float)]
    return pd.DataFrame({"y": values}, index=idx)


def _small_panel_X():
    idx = pd.MultiIndex.from_product(
        [["a", "b"], pd.RangeIndex(6)], names=["id", "time"]
    )
    x0 = np.r_[np.arange(6, dtype=float), np.arange(10, 16, dtype=float)] * 10
    x1 = x0 + 1000
    return pd.DataFrame({"x0": x0, "x1": x1}, index=idx)


def _small_multivariate_panel_series():
    y = _small_panel_series()
    y["z"] = y["y"] + 100.0
    return y


def _unequal_panel_series():
    short_idx = pd.MultiIndex.from_product(
        [["short"], pd.RangeIndex(6)], names=["id", "time"]
    )
    long_idx = pd.MultiIndex.from_product(
        [["long"], pd.RangeIndex(8)], names=["id", "time"]
    )
    short = pd.DataFrame({"y": np.arange(6, dtype=float)}, index=short_idx)
    long = pd.DataFrame({"y": np.arange(10, 18, dtype=float)}, index=long_idx)
    return pd.concat([short, long])


def test_window_normalizers_are_skbase_objects_and_invert_values():
    """Window normalizers implement the skbase object contract."""
    lags = np.array([2.0, 4.0, 6.0])

    for normalizer in [
        MeanWindowNormalizer(),
        SubtractMeanNormalizer(),
        ZScoreWindowNormalizer(),
        MinMaxWindowNormalizer(),
    ]:
        assert isinstance(normalizer, BaseWindowNormalizer)
        assert isinstance(normalizer, BaseObject)
        assert normalizer.get_tag("object_type") == "window-normalizer"
        assert scitype(normalizer) == "window-normalizer"
        lags_t, target_t = normalizer.transform(lags, 8.0)
        assert lags_t.shape == lags.shape
        assert normalizer.inverse_transform(target_t, lags) == pytest.approx(8.0)


def test_window_normalizers_batch_methods_match_scalar_contract():
    """Batch normalizer methods match scalar transform semantics."""
    lags = np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0], [5.0, 7.0, 11.0]])
    target = np.array([8.0, 9.0, 13.0])

    for normalizer in [
        MeanWindowNormalizer(),
        SubtractMeanNormalizer(),
        ZScoreWindowNormalizer(),
        MinMaxWindowNormalizer(),
    ]:
        batch_lags_t, batch_target_t = normalizer.batch_transform(lags, target)
        expected_lags_t = []
        expected_target_t = []

        for lag_row, target_value in zip(lags, target):
            lag_row_t, target_value_t = normalizer.transform(lag_row, target_value)
            expected_lags_t.append(lag_row_t)
            expected_target_t.append(target_value_t)

        np.testing.assert_allclose(batch_lags_t, np.asarray(expected_lags_t))
        np.testing.assert_allclose(batch_target_t, np.asarray(expected_target_t))

        recovered = normalizer.batch_inverse_transform(batch_target_t, lags)
        np.testing.assert_allclose(recovered, target)

        batch_lags_only, batch_target_none = normalizer.batch_transform(lags)
        assert batch_target_none is None
        np.testing.assert_allclose(batch_lags_only, np.asarray(expected_lags_t))


def test_reduction_forecaster_canonical_and_compatibility_imports_match():
    """Canonical compose import and old base import resolve to the same class."""
    from sktime.forecasting.compose import ReductionForecaster as BaseReduction
    from sktime.forecasting.compose._pretrain_reduce import (
        ReductionForecaster as ComposeReduction,
    )

    assert BaseReduction is ComposeReduction


@pytest.mark.parametrize(
    "step, expected_X, expected_y",
    [
        (
            1,
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0],
                [11.0, 12.0, 13.0],
                [12.0, 13.0, 14.0],
            ],
            [3.0, 4.0, 5.0, 13.0, 14.0, 15.0],
        ),
        (
            2,
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [10.0, 11.0, 12.0],
                [11.0, 12.0, 13.0],
            ],
            [4.0, 5.0, 14.0, 15.0],
        ),
        (
            3,
            [
                [0.0, 1.0, 2.0],
                [10.0, 11.0, 12.0],
            ],
            [5.0, 15.0],
        ),
    ],
)
def test_build_supervised_table_matches_direct_horizon_rows(
    step, expected_X, expected_y
):
    """Supervised rows use per-instance lag windows and horizon-specific targets."""
    Xt, yt = _build_supervised_table(
        y=_small_panel_series(),
        X=None,
        window_length=3,
        steps_ahead=step,
        normalizer=None,
    )

    np.testing.assert_array_equal(Xt, np.asarray(expected_X, dtype=float))
    np.testing.assert_array_equal(yt, np.asarray(expected_y, dtype=float))


def test_build_supervised_table_appends_X_at_target_timestamp():
    """Exogenous rows are taken from the target timestamp, not the lag end."""
    Xt, yt = _build_supervised_table(
        y=_small_panel_series(),
        X=_small_panel_X(),
        window_length=3,
        steps_ahead=2,
        normalizer=None,
    )

    expected_X = np.array(
        [
            [0.0, 1.0, 2.0, 40.0, 1040.0],
            [1.0, 2.0, 3.0, 50.0, 1050.0],
            [10.0, 11.0, 12.0, 140.0, 1140.0],
            [11.0, 12.0, 13.0, 150.0, 1150.0],
        ]
    )
    expected_y = np.array([4.0, 5.0, 14.0, 15.0])

    np.testing.assert_array_equal(Xt, expected_X)
    np.testing.assert_array_equal(yt, expected_y)


def test_build_supervised_table_rejects_missing_X_target_timestamp():
    """All target timestamps used for supervised rows must be available in X."""
    X = _small_panel_X().drop(("a", 4))

    with pytest.raises(ValueError, match="supervised target timestamps"):
        _build_supervised_table(
            y=_small_panel_series(),
            X=X,
            window_length=3,
            steps_ahead=2,
            normalizer=None,
        )


@pytest.mark.parametrize(
    "normalization_strategy",
    [None, "mean", "subtract_mean", "zscore", "minmax"],
)
def test_build_supervised_table_matches_rowwise_normalization(
    normalization_strategy,
):
    """Built supervised matrices preserve row-wise normalizer semantics."""
    y = pd.Series(
        [-1.0, 0.0, 1.0, 3.0, 6.0, 10.0],
        index=pd.RangeIndex(6),
        name="y",
    )
    normalizer = _resolve_normalizer(normalization_strategy)

    Xt, yt = _build_supervised_table(
        y=y,
        X=None,
        window_length=3,
        steps_ahead=2,
        normalizer=normalizer,
    )

    expected_X = []
    expected_y = []
    for end in range(3, 5):
        lags = y.to_numpy(dtype=float)[end - 3 : end]
        target = y.to_numpy(dtype=float)[end + 1]
        if normalizer is not None:
            lags, target = normalizer.transform(lags, target)
        expected_X.append(lags)
        expected_y.append(target)

    np.testing.assert_allclose(Xt, np.asarray(expected_X, dtype=float))
    np.testing.assert_allclose(yt, np.asarray(expected_y, dtype=float))


def test_custom_normalizer_preserves_rowwise_call_order():
    """Custom normalizers without batch overrides keep current call order."""
    LoggingNormalizer.calls = []
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
        normalization_strategy=LoggingNormalizer(),
    )

    forecaster.pretrain(_small_panel_series())

    expected = [
        ((0.0, 1.0, 2.0), 3.0),
        ((1.0, 2.0, 3.0), 4.0),
        ((2.0, 3.0, 4.0), 5.0),
        ((10.0, 11.0, 12.0), 13.0),
        ((11.0, 12.0, 13.0), 14.0),
        ((12.0, 13.0, 14.0), 15.0),
        ((0.0, 1.0, 2.0), 4.0),
        ((1.0, 2.0, 3.0), 5.0),
        ((10.0, 11.0, 12.0), 14.0),
        ((11.0, 12.0, 13.0), 15.0),
    ]
    assert LoggingNormalizer.calls == expected


def test_supervised_table_uses_custom_batch_transform_hook():
    """Custom batch_transform hooks are used without scalar fallback calls."""
    BatchHookNormalizer.reset()

    Xt, yt = _build_supervised_table(
        y=_small_panel_series(),
        X=None,
        window_length=3,
        steps_ahead=2,
        normalizer=BatchHookNormalizer(),
    )

    expected_X = np.array(
        [
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0],
            [21.0, 22.0, 23.0],
        ]
    )
    expected_y = np.array([104.0, 105.0, 114.0, 115.0])

    np.testing.assert_array_equal(Xt, expected_X)
    np.testing.assert_array_equal(yt, expected_y)
    assert len(BatchHookNormalizer.batch_transform_targets) == 2
    assert BatchHookNormalizer.scalar_transform_calls == 0
    assert BatchHookNormalizer.scalar_inverse_calls == 0


def test_prediction_uses_custom_batch_normalizer_hooks():
    """Prediction row normalization and inverse scaling use batch hooks."""
    BatchHookNormalizer.reset()
    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=2.0),
        window_length=3,
        steps_ahead=1,
        normalization_strategy=BatchHookNormalizer(),
    )

    forecaster.fit(y, fh=[1])
    y_pred = forecaster.predict()

    assert y_pred.iloc[0] == pytest.approx(7.0)
    assert BatchHookNormalizer.batch_transform_targets[-1] is None
    assert BatchHookNormalizer.batch_inverse_calls == 1
    assert BatchHookNormalizer.scalar_transform_calls == 0
    assert BatchHookNormalizer.scalar_inverse_calls == 0


def test_pretrain_fits_direct_heads_and_tracks_pretrained_params():
    """Pretrain learns pooled direct heads through BaseForecaster.pretrain."""
    RecordingRegressor.total_fit_calls = 0

    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
        normalization_strategy="mean",
    )

    forecaster.pretrain(_panel_series())

    assert forecaster.state == "pretrained"
    assert RecordingRegressor.total_fit_calls == 2
    assert len(forecaster.direct_estimators_) == 2

    pretrained = forecaster.get_pretrained_params()
    assert "direct_estimators_" in pretrained
    assert "n_pretrain_instances_" in pretrained
    assert pretrained["n_pretrain_instances_"] == 2


def test_pretrain_with_exogenous_variables_uses_X_in_direct_heads():
    """Pretrain appends exogenous target-time rows to each direct head."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )

    forecaster.pretrain(_small_panel_series(), X=_small_panel_X())

    assert forecaster.x_columns_ == ["x0", "x1"]
    assert [est.fit_shape_ for est in forecaster.direct_estimators_] == [
        (6, 5),
        (4, 5),
    ]
    expected_step_2_X = np.array(
        [
            [0.0, 1.0, 2.0, 40.0, 1040.0],
            [1.0, 2.0, 3.0, 50.0, 1050.0],
            [10.0, 11.0, 12.0, 140.0, 1140.0],
            [11.0, 12.0, 13.0, 150.0, 1150.0],
        ]
    )
    np.testing.assert_array_equal(
        forecaster.direct_estimators_[1].X_,
        expected_step_2_X,
    )


def test_pretrain_pools_multivariate_panel_columns_as_univariate_series():
    """Multivariate pretrain panels are split into pooled univariate series."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )

    forecaster.pretrain(_small_multivariate_panel_series(), X=_small_panel_X())

    assert forecaster.n_pretrain_instances_ == 4
    assert forecaster.n_pretrain_timepoints_ == 24
    assert [est.fit_shape_ for est in forecaster.direct_estimators_] == [
        (12, 5),
        (8, 5),
    ]


def test_pretrain_with_normalizer_instance_does_not_normalize_X():
    """Window normalizers transform y lags and targets, but not exogenous rows."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
        normalization_strategy=SubtractMeanNormalizer(),
    )

    forecaster.pretrain(_small_panel_series(), X=_small_panel_X())

    step_2_X = forecaster.direct_estimators_[1].X_
    np.testing.assert_array_equal(
        step_2_X,
        np.array(
            [
                [-1.0, 0.0, 1.0, 40.0, 1040.0],
                [-1.0, 0.0, 1.0, 50.0, 1050.0],
                [-1.0, 0.0, 1.0, 140.0, 1140.0],
                [-1.0, 0.0, 1.0, 150.0, 1150.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        step_2_X[:, -2:],
        _small_panel_X().iloc[[4, 5, 10, 11]].to_numpy(dtype=float),
    )


def test_fit_after_pretrain_with_new_y_X_keeps_future_X_unnormalized():
    """Local fit y is normalized for prediction, but future X remains raw."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=0.0),
        window_length=3,
        steps_ahead=1,
        normalization_strategy=SubtractMeanNormalizer(),
    )
    forecaster.pretrain(_small_panel_series(), X=_small_panel_X())

    y_fit = pd.Series([20.0, 21.0, 22.0, 30.0], index=pd.RangeIndex(4), name="y")
    X_fit = pd.DataFrame(
        {
            "x0": [200.0, 210.0, 220.0, 300.0],
            "x1": [1200.0, 1210.0, 1220.0, 1300.0],
        },
        index=y_fit.index,
    )
    X_pred = pd.DataFrame(
        {"x0": [400.0, 500.0], "x1": [1400.0, 1500.0]},
        index=pd.RangeIndex(4, 6),
    )
    forecaster.fit(y_fit, X=X_fit, fh=[1, 2])

    RecordingRegressor.predict_log = []
    forecaster.predict(X=X_pred)

    logged_rows = [entry[2].ravel() for entry in RecordingRegressor.predict_log]
    np.testing.assert_allclose(
        logged_rows[0],
        np.array(
            [
                -3.333333333333332,
                -2.333333333333332,
                5.666666666666668,
                400.0,
                1400.0,
            ]
        ),
    )
    np.testing.assert_array_equal(
        logged_rows[0][-2:],
        X_pred.iloc[0].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        logged_rows[1][-2:],
        X_pred.iloc[1].to_numpy(dtype=float),
    )


def test_pretrain_without_exogenous_variables_uses_lag_features_only():
    """Pretrain without X keeps reducer rows to the lag-window width."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=1.0),
        window_length=3,
        steps_ahead=2,
    )

    forecaster.pretrain(_small_panel_series())

    assert forecaster.x_columns_ is None
    assert [est.fit_shape_ for est in forecaster.direct_estimators_] == [
        (6, 3),
        (4, 3),
    ]

    y_local = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    forecaster.fit(y_local, fh=[1, 2])

    RecordingRegressor.predict_log = []
    forecaster.predict()

    assert [entry[2].shape[1] for entry in RecordingRegressor.predict_log] == [3, 3]


def test_pretrain_handles_unequal_length_input_series():
    """Pretrain pools unequal-length panels without crossing instance boundaries."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )

    forecaster.pretrain(_unequal_panel_series())

    assert [est.fit_shape_ for est in forecaster.direct_estimators_] == [
        (8, 3),
        (6, 3),
    ]
    expected_step_2_X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
            [12.0, 13.0, 14.0],
            [13.0, 14.0, 15.0],
        ]
    )
    expected_step_2_y = np.array([4.0, 5.0, 14.0, 15.0, 16.0, 17.0])
    np.testing.assert_array_equal(
        forecaster.direct_estimators_[1].X_,
        expected_step_2_X,
    )
    np.testing.assert_array_equal(
        forecaster.direct_estimators_[1].y_,
        expected_step_2_y,
    )


def test_fit_after_pretrain_preserves_global_heads():
    """Fit stores local context without rebuilding pretrained heads."""
    RecordingRegressor.total_fit_calls = 0
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series())
    heads_before = forecaster.direct_estimators_
    fit_calls_after_pretrain = RecordingRegressor.total_fit_calls

    y_local = pd.Series(
        np.arange(20, 28, dtype=float), index=pd.RangeIndex(8), name="y"
    )
    forecaster.fit(y_local, fh=[1, 2, 3])

    assert forecaster.state == "fitted"
    assert forecaster.direct_estimators_ is heads_before
    assert RecordingRegressor.total_fit_calls == fit_calls_after_pretrain
    assert np.array_equal(forecaster.last_window_, np.array([25.0, 26.0, 27.0]))


def test_clone_after_pretrain_preserves_pretrained_heads_only():
    """Clone preserves pretrained heads but not local fitted context."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series())
    forecaster.fit(
        pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y"),
        fh=[1, 2],
    )

    cloned = forecaster.clone()

    assert cloned.state == "pretrained"
    assert hasattr(cloned, "direct_estimators_")
    assert not hasattr(cloned, "last_window_")


def test_fitted_params_exclude_pretrained_heads_after_fit():
    """Pretrained params stay out of fitted parameter namespace."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series())
    forecaster.fit(
        pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y"),
        fh=[1, 2],
    )

    fitted = forecaster.get_fitted_params()

    assert "direct_estimators" not in fitted
    assert "one_step_estimator" not in fitted
    assert "normalizer" not in fitted
    assert "last_window" in fitted


def test_pretrain_update_after_fit_does_not_capture_local_context():
    """Incremental pretrain after fit keeps local fit attrs out of pretrain params."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series())
    forecaster.fit(
        pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y"),
        fh=[1, 2],
    )

    forecaster.pretrain(_panel_series())
    pretrained = forecaster.get_pretrained_params()

    assert forecaster.state == "pretrained"
    assert "last_window_" not in pretrained
    assert "train_index_" not in pretrained
    assert "y_was_dataframe_" not in pretrained
    assert "y_name_" not in pretrained


def test_predict_returns_relative_horizon_index_after_pretrain_and_fit():
    """Predict uses local fit context and returns sktime relative fh index."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=1.5),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series())
    y_local = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    forecaster.fit(y_local, fh=[1, 2, 3])

    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.Series)
    assert list(y_pred.index) == [8, 9, 10]
    assert list(y_pred.values) == [1.5, 1.5, 1.5]


def test_prediction_reuses_direct_heads_in_recursive_blocks():
    """Forecasts beyond K reuse all direct heads in recursive blocks."""
    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.fit(y, fh=[1, 2, 3, 4, 5])

    forecaster.direct_estimators_[0].constant = 10.0
    forecaster.direct_estimators_[0].label = "one"
    forecaster.direct_estimators_[1].constant = 20.0
    forecaster.direct_estimators_[1].label = "two"

    RecordingRegressor.predict_log = []
    y_pred = forecaster.predict()

    assert list(y_pred.values) == [10.0, 20.0, 10.0, 20.0, 10.0]
    assert [entry[0] for entry in RecordingRegressor.predict_log] == [
        "one",
        "two",
        "one",
        "two",
        "one",
    ]
    logged_rows = [
        entry[2].ravel().tolist() for entry in RecordingRegressor.predict_log
    ]
    assert logged_rows == [
        [5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0],
        [7.0, 10.0, 20.0],
        [7.0, 10.0, 20.0],
        [20.0, 10.0, 20.0],
    ]


def test_predict_preserves_single_column_dataframe_output():
    """A single-column DataFrame y produces a single-column DataFrame forecast."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=2.0),
        window_length=3,
        steps_ahead=1,
    )
    y = pd.DataFrame({"target": np.arange(8, dtype=float)}, index=pd.RangeIndex(8))

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.DataFrame)
    assert list(y_pred.columns) == ["target"]
    assert list(y_pred.index) == [8, 9]
    assert list(y_pred["target"]) == [2.0, 2.0]


def test_vectorized_hierarchical_predict_slices_future_X_by_group():
    """Vectorized hierarchical predict passes each clone its own future X rows."""
    time = pd.period_range("2000-01", periods=7, freq="M")
    train_index = pd.MultiIndex.from_product(
        [["h0", "h1"], ["a", "b"], time[:5]],
        names=["level_0", "level_1", "time"],
    )
    full_x_index = pd.MultiIndex.from_product(
        [["h0", "h1"], ["a", "b"], time],
        names=["level_0", "level_1", "time"],
    )
    pred_index = pd.MultiIndex.from_product(
        [["h0", "h1"], ["a", "b"], time[5:]],
        names=["level_0", "level_1", "time"],
    )
    y_train = pd.DataFrame(
        {"y": np.arange(len(train_index), dtype=float)},
        index=train_index,
    )
    X = pd.DataFrame(
        {"x": np.arange(len(full_x_index), dtype=float)},
        index=full_x_index,
    )
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=1.0),
        window_length=3,
        steps_ahead=2,
    )

    forecaster.fit(y_train, X=X.loc[train_index], fh=[1, 2])
    RecordingRegressor.predict_log = []
    y_pred = forecaster.predict(X=X.loc[pred_index])

    assert y_pred.index.equals(pred_index)
    logged_future_x = [entry[2][0, -1] for entry in RecordingRegressor.predict_log]
    np.testing.assert_array_equal(
        logged_future_x,
        X.loc[pred_index, "x"].to_numpy(dtype=float),
    )


def test_predict_appends_future_exogenous_rows():
    """Future X rows are appended to lag features at prediction time."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(constant=1.0),
        window_length=3,
        steps_ahead=1,
    )
    forecaster.pretrain(_panel_series(), X=_panel_X())

    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    X_fit = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=y.index)
    X_pred = pd.DataFrame({"x": [1000.0, 1001.0]}, index=pd.RangeIndex(8, 10))
    forecaster.fit(y, X=X_fit, fh=[1, 2])

    RecordingRegressor.predict_log = []
    forecaster.predict(X=X_pred)

    first_row = RecordingRegressor.predict_log[0][2]
    second_row = RecordingRegressor.predict_log[1][2]
    assert first_row.shape[1] == 4
    assert first_row[0, -1] == 1000.0
    assert second_row[0, -1] == 1001.0


def test_block_recursive_prediction_aligns_future_X_by_absolute_step():
    """Recursive blocks use the matching future X row for each horizon step."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
    )
    forecaster.pretrain(_panel_series(), X=_panel_X())

    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    X_fit = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=y.index)
    X_pred = pd.DataFrame(
        {"x": [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]},
        index=pd.RangeIndex(8, 13),
    )
    forecaster.fit(y, X=X_fit, fh=[1, 2, 3, 4, 5])

    forecaster.direct_estimators_[0].constant = 10.0
    forecaster.direct_estimators_[0].label = "one"
    forecaster.direct_estimators_[1].constant = 20.0
    forecaster.direct_estimators_[1].label = "two"

    RecordingRegressor.predict_log = []
    forecaster.predict(X=X_pred)

    assert [entry[0] for entry in RecordingRegressor.predict_log] == [
        "one",
        "two",
        "one",
        "two",
        "one",
    ]
    logged_rows = [
        entry[2].ravel().tolist() for entry in RecordingRegressor.predict_log
    ]
    assert logged_rows == [
        [5.0, 6.0, 7.0, 1000.0],
        [5.0, 6.0, 7.0, 1001.0],
        [7.0, 10.0, 20.0, 1002.0],
        [7.0, 10.0, 20.0, 1003.0],
        [20.0, 10.0, 20.0, 1004.0],
    ]


def test_block_recursive_prediction_normalizes_each_block_context_not_X():
    """Later block heads share normalized block-start lags and raw future X."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=2,
        normalization_strategy=SubtractMeanNormalizer(),
    )
    forecaster.pretrain(_panel_series(), X=_panel_X())

    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    X_fit = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=y.index)
    X_pred = pd.DataFrame(
        {"x": [1000.0, 1001.0, 1002.0, 1003.0]},
        index=pd.RangeIndex(8, 12),
    )
    forecaster.fit(y, X=X_fit, fh=[1, 2, 3, 4])

    forecaster.direct_estimators_[0].constant = 10.0
    forecaster.direct_estimators_[0].label = "one"
    forecaster.direct_estimators_[1].constant = 20.0
    forecaster.direct_estimators_[1].label = "two"

    RecordingRegressor.predict_log = []
    y_pred = forecaster.predict(X=X_pred)

    np.testing.assert_allclose(
        y_pred.to_numpy(),
        np.array([16.0, 26.0, 26.333333333333332, 36.33333333333333]),
    )
    logged_rows = [entry[2].ravel() for entry in RecordingRegressor.predict_log]
    np.testing.assert_allclose(
        logged_rows,
        np.array(
            [
                [-1.0, 0.0, 1.0, 1000.0],
                [-1.0, 0.0, 1.0, 1001.0],
                [
                    -9.333333333333332,
                    -0.33333333333333215,
                    9.666666666666668,
                    1002.0,
                ],
                [
                    -9.333333333333332,
                    -0.33333333333333215,
                    9.666666666666668,
                    1003.0,
                ],
            ]
        ),
    )


def test_fit_rejects_new_X_when_pretrained_without_X():
    """Fit cannot add exogenous features to heads pretrained without them."""
    forecaster = ReductionForecaster(
        estimator=RecordingRegressor(),
        window_length=3,
        steps_ahead=1,
    )
    forecaster.pretrain(_panel_series())

    y = pd.Series(np.arange(8, dtype=float), index=pd.RangeIndex(8), name="y")
    X = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=y.index)

    with pytest.raises(ValueError, match="pretrained without X"):
        forecaster.fit(y, X=X, fh=[1, 2])
