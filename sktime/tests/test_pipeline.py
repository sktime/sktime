"""Composition-level integration tests for sktime forecasting pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone


def _make_univariate(n=48, freq="M", start="2019-01-01", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(rng.standard_normal(n), index=idx, name="y")


def _make_multivariate(n=48, freq="M", start="2019-01-01", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"y1": rng.standard_normal(n), "y2": rng.standard_normal(n)},
        index=idx,
    )


FH = [1, 2, 3]


@pytest.mark.composition
class TestTransformedTargetForecaster:

    def test_detrend_then_naive_fit_predict(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        y = _make_univariate()
        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        assert isinstance(pred, pd.Series)
        assert len(pred) == len(FH)
        assert not pred.isna().any()

    def test_boxcox_then_naive_fit_predict(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        y = _make_univariate() + 10
        pipe = TransformedTargetForecaster(
            steps=[
                ("boxcox", BoxCoxTransformer()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        assert len(pred) == len(FH)
        assert not pred.isna().any()

    def test_multiple_transformers_chained(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender, Deseasonalizer

        y = _make_univariate(n=48).to_period("M")
        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("deseason", Deseasonalizer(sp=12)),
                ("forecast", NaiveForecaster(strategy="mean")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        assert len(pred) == len(FH)
        assert not pred.isna().any()

    def test_predict_returns_correct_index(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        y = _make_univariate(n=36, freq="M", start="2020-01-01")
        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        last_train = y.index[-1]

        assert isinstance(pred.index, pd.DatetimeIndex)
        assert (pred.index > last_train).all()
        if pred.index.freq is not None:
            assert pred.index.freq == y.index.freq
        assert len(pred) == len(FH)


@pytest.mark.composition
class TestCloneCorrectness:

    def test_naive_forecaster_clone_is_unfitted(self):
        from sktime.forecasting.naive import NaiveForecaster

        y = _make_univariate()
        original = NaiveForecaster(strategy="last")
        original.fit(y, fh=FH)

        cloned = clone(original)

        assert not cloned.is_fitted

    def test_clone_preserves_params(self):
        from sktime.forecasting.naive import NaiveForecaster

        original = NaiveForecaster(strategy="mean", sp=12, window_length=6)
        cloned = clone(original)

        assert original.get_params() == cloned.get_params()

    def test_pipeline_clone_is_unfitted(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        y = _make_univariate()
        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        cloned = clone(pipe)

        assert not cloned.is_fitted

    def test_clone_produces_separate_object(self):
        from sktime.forecasting.naive import NaiveForecaster

        original = NaiveForecaster(strategy="last")
        cloned = clone(original)

        assert cloned is not original

    def test_clone_deep_isolation_pipeline_steps(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        cloned = clone(pipe)

        cloned.set_params(forecast__strategy="mean")

        original_strategy = pipe.get_params()["forecast__strategy"]
        assert original_strategy == "last"

    def test_grid_search_uses_clone_correctly(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.model_selection import ForecastingGridSearchCV
        from sktime.split import SingleWindowSplitter

        y = _make_univariate(n=60)
        cv = SingleWindowSplitter(fh=FH)
        param_grid = {"strategy": ["last", "mean"]}

        gscv = ForecastingGridSearchCV(
            forecaster=NaiveForecaster(strategy="last"),
            cv=cv,
            param_grid=param_grid,
            scoring=None,
        )
        gscv.fit(y, fh=FH)
        pred = gscv.predict(fh=FH)

        assert len(pred) == len(FH)
        assert gscv.best_params_ is not None
        assert "strategy" in gscv.best_params_
        assert gscv.best_params_["strategy"] in param_grid["strategy"]
        assert len(gscv.cv_results_) > 0


@pytest.mark.composition
class TestTransformerInverseInPipeline:

    def test_detrender_inverse_recovers_original_scale(self):
        from sktime.transformations.series.detrend import Detrender

        y = pd.Series(
            np.arange(1.0, 49.0),
            index=pd.date_range("2020", periods=48, freq="M"),
        )
        t = Detrender()
        y_transformed = t.fit_transform(y)
        y_recovered = t.inverse_transform(y_transformed)

        pd.testing.assert_series_equal(
            y,
            y_recovered,
            check_names=False,
            atol=1e-6,
        )

    def test_deseasonalizer_inverse_recovers_original_scale(self):
        from sktime.transformations.series.detrend import Deseasonalizer

        rng = np.random.default_rng(0)
        seasonal = np.tile(np.sin(np.linspace(0, 2 * np.pi, 12)), 4)
        y = pd.Series(
            rng.standard_normal(48) + seasonal,
            index=pd.period_range("2020", periods=48, freq="M"),
        )
        t = Deseasonalizer(sp=12)
        y_transformed = t.fit_transform(y)
        y_recovered = t.inverse_transform(y_transformed)

        pd.testing.assert_series_equal(
            y,
            y_recovered,
            check_names=False,
            atol=1e-4,
        )

    def test_pipeline_predict_is_in_original_scale(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        y = pd.Series(
            np.arange(50.0, 98.0),
            index=pd.date_range("2020", periods=48, freq="M"),
        )
        pipe = TransformedTargetForecaster(
            steps=[
                ("detrend", Detrender()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        assert (pred > y.min()).all()

    def test_boxcox_inverse_inside_pipeline(self):
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        y = _make_univariate() + 10
        pipe = TransformedTargetForecaster(
            steps=[
                ("boxcox", BoxCoxTransformer()),
                ("forecast", NaiveForecaster(strategy="last")),
            ]
        )
        pipe.fit(y, fh=FH)
        pred = pipe.predict()

        assert (pred > y.min()).all()
        assert not pred.isna().any()


@pytest.mark.composition
class TestColumnEnsembleForecaster:

    def test_fit_predict_multivariate(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.compose import ColumnEnsembleForecaster

        y = _make_multivariate(n=36)
        forecaster = ColumnEnsembleForecaster(
            forecasters=[
                ("f1", NaiveForecaster(strategy="last"), "y1"),
                ("f2", NaiveForecaster(strategy="mean"), "y2"),
            ]
        )
        forecaster.fit(y, fh=FH)
        pred = forecaster.predict()

        assert isinstance(pred, pd.DataFrame)
        assert list(pred.columns) == ["y1", "y2"]
        assert len(pred) == len(FH)
        assert not pred.isna().any().any()

    def test_clone_preserves_column_assignments(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.compose import ColumnEnsembleForecaster

        forecaster = ColumnEnsembleForecaster(
            forecasters=[
                ("f1", NaiveForecaster(strategy="last"), "y1"),
                ("f2", NaiveForecaster(strategy="mean"), "y2"),
            ]
        )
        cloned = clone(forecaster)

        orig = [(name, col) for name, _, col in forecaster.forecasters]
        cloned_ = [(name, col) for name, _, col in cloned.forecasters]
        assert orig == cloned_

    def test_fit_predict_broadcast_single_forecaster(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.compose import ColumnEnsembleForecaster

        y = _make_multivariate(n=36)
        forecaster = ColumnEnsembleForecaster(
            forecasters=NaiveForecaster(strategy="last")
        )
        forecaster.fit(y, fh=FH)
        pred = forecaster.predict()

        assert isinstance(pred, pd.DataFrame)
        assert len(pred) == len(FH)

    def test_output_columns_match_trained_columns(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.compose import ColumnEnsembleForecaster

        y = _make_multivariate(n=36)
        forecaster = ColumnEnsembleForecaster(
            forecasters=[
                ("f1", NaiveForecaster(strategy="last"), "y1"),
                ("f2", NaiveForecaster(strategy="mean"), "y2"),
            ]
        )
        forecaster.fit(y, fh=FH)
        pred = forecaster.predict()

        assert list(pred.columns) == ["y1", "y2"]

    def test_column_order_mismatch_produces_correct_output(self):
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.compose import ColumnEnsembleForecaster

        rng = np.random.default_rng(7)
        idx = pd.date_range("2020", periods=36, freq="M")

        y = pd.DataFrame(
            {
                "y2": rng.standard_normal(36) + 1,
                "y1": rng.standard_normal(36) + 100,
            },
            index=idx,
        )

        forecaster = ColumnEnsembleForecaster(
            forecasters=[
                ("f1", NaiveForecaster(strategy="last"), "y1"),
                ("f2", NaiveForecaster(strategy="last"), "y2"),
            ]
        )
        forecaster.fit(y, fh=[1])
        pred = forecaster.predict()

        assert pred["y1"].iloc[-1] == pytest.approx(y["y1"].iloc[-1], rel=1e-8)
        assert pred["y2"].iloc[-1] == pytest.approx(y["y2"].iloc[-1], rel=1e-8)