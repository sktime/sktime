"""Tests for TinyTimeMixerForecaster."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from types import SimpleNamespace

import pandas as pd
import pytest

from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_basic_functionality():
    """Test basic forecaster functionality without few-shot learning."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 8, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict()

    # Basic assertions
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_exogenous_variables():
    """Test forecaster with exogenous variables."""
    from sktime.datasets import load_longley
    from sktime.forecasting.ttm import TinyTimeMixerForecaster
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, _, X_train, X_future = temporal_train_test_split(y, X, test_size=2)

    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 6, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y_train, X=X_train, fh=[1, 2])
    y_pred = forecaster.predict(X=X_future)

    # Basic assertions
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_few_shot_sampling():
    """Test few-shot sampling functionality."""
    forecaster = TinyTimeMixerForecaster()
    config = SimpleNamespace(context_length=2, prediction_length=1)

    index = pd.RangeIndex(start=0, stop=10, step=1)
    y = pd.Series(range(10), index=index, name="target")
    X = pd.DataFrame(
        {"feat_1": range(10), "feat_2": range(10, 20)},
        index=index,
    )

    y_sampled, X_sampled = forecaster._apply_few_shot_sampling(
        y_train=y,
        X_train=X,
        ratio=0.5,
        config=config,
    )

    expected_y = y.tail(4)
    expected_X = X.tail(4)

    assert y_sampled.equals(expected_y)
    assert X_sampled.equals(expected_X)

    config_warn = SimpleNamespace(context_length=4, prediction_length=2)
    y_short = pd.Series(range(8), index=pd.RangeIndex(0, 8), name="target")

    with pytest.warns(UserWarning):
        y_sampled_warn, X_sampled_warn = forecaster._apply_few_shot_sampling(
            y_train=y_short,
            X_train=None,
            ratio=0.1,
            config=config_warn,
        )

    expected_y_warn = y_short.tail(6)
    assert y_sampled_warn.equals(expected_y_warn)
    assert X_sampled_warn is None

    config_multi = SimpleNamespace(context_length=2, prediction_length=1)
    series_ids = ["series_a", "series_b"]
    time_index = pd.RangeIndex(0, 6)
    multi_index = pd.MultiIndex.from_product(
        [series_ids, time_index], names=["series", "time"]
    )

    y_multi = pd.Series(range(12), index=multi_index, name="target")
    X_multi = pd.DataFrame(
        {
            "feat_1": range(100, 112),
            "feat_2": range(200, 212),
        },
        index=multi_index,
    )

    y_sampled_multi, X_sampled_multi = forecaster._apply_few_shot_sampling(
        y_train=y_multi,
        X_train=X_multi,
        ratio=0.5,
        config=config_multi,
    )

    expected_y_multi = pd.concat(
        [y_multi.loc["series_a"].tail(3), y_multi.loc["series_b"].tail(3)]
    )
    expected_X_multi = pd.concat(
        [X_multi.loc["series_a"].tail(3), X_multi.loc["series_b"].tail(3)]
    )

    assert y_sampled_multi.equals(expected_y_multi)
    assert X_sampled_multi.equals(expected_X_multi)


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_backward_compatibility():
    """Test that existing code works without few-shot parameters."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    # Test original usage pattern (no few-shot parameters)
    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 8, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict()

    # Verify backward compatibility
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_few_shot_capability():
    """Test few-shot learning functionality with different ratios."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    # Test different few-shot ratios
    ratios = [0.2, 0.5, 0.8]

    for ratio in ratios:
        forecaster = TinyTimeMixerForecaster(
            model_path=None,
            fit_strategy="full",
            few_shot_ratio=ratio,
            config={"context_length": 8, "prediction_length": 2},
            training_args={
                "max_steps": 2,
                "output_dir": "test_output",
                "per_device_train_batch_size": 4,
                "report_to": "none",
            },
        )

        forecaster.fit(y, fh=[1, 2])
        y_pred = forecaster.predict()

        # Verify few-shot functionality
        assert y_pred is not None
        assert len(y_pred) == 2
        assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ttm_estimator_check():
    """Run standard estimator checks for TTM."""
    from sktime.forecasting.ttm import TinyTimeMixerForecaster
    from sktime.utils import check_estimator

    check_estimator(TinyTimeMixerForecaster, raise_exceptions=True)
