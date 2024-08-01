# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for interfacing estimators from neuralforecast."""

import pandas
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.neuralforecast import NeuralForecastLSTM, NeuralForecastRNN
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

__author__ = ["yarnabrina", "pranavvp16", "geetu040"]

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_univariate_y_without_X(model_class) -> None:
    """Test with single endogenous without exogenous."""
    # define model
    model = model_class(freq="A-DEC", max_steps=5, trainer_kwargs={"logger": False})

    # attempt fit with negative fh
    with pytest.raises(NotImplementedError):
        model.fit(y_train, fh=[-2, -1, 0, 1, 2])

    # train model
    model.fit(y_train, fh=[1, 2, 3, 4])

    # predict with trained model
    y_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(y_pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_univariate_y_with_X(model_class) -> None:
    """Test with single endogenous with exogenous."""
    # select feature columns
    exog_list = ["GNPDEFL", "GNP", "UNEMP"]

    # define model
    model = model_class(
        freq="A-DEC",
        futr_exog_list=exog_list,
        max_steps=5,
        trainer_kwargs={"logger": False},
    )

    # attempt fit without X
    with pytest.raises(
        ValueError, match="Missing exogeneous data, 'futr_exog_list' is non-empty."
    ):
        model.fit(y_train, fh=[1, 2, 3, 4])

    # train model with all X columns
    model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])

    # attempt predict without X
    with pytest.raises(
        ValueError, match="Missing exogeneous data, 'futr_exog_list' is non-empty."
    ):
        model.predict()

    # predict with only selected columns
    # checking that rest are not used
    y_pred = model.predict(X=X_test[exog_list])

    # check prediction index
    pandas.testing.assert_index_equal(y_pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_multivariate_y_without_X(model_class) -> None:
    """Test with multiple endogenous without exogenous."""
    # define model
    model = model_class(freq="A-DEC", max_steps=5, trainer_kwargs={"logger": False})

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # predict with trained model
    X_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(X_pred.index, X_test.index, check_names=False)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_non_default_loss(model_class) -> None:
    """Test with multiple endogenous without exogenous."""
    # import non-default pytorch losses
    from neuralforecast.losses.pytorch import MASE, HuberQLoss

    # define model
    model = model_class(
        freq="A-DEC",
        loss=HuberQLoss(0.5),
        valid_loss=MASE(1),
        max_steps=5,
        trainer_kwargs={"logger": False},
    )

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # predict with trained model
    X_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(X_pred.index, X_test.index, check_names=False)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_fail_with_multiple_predictions(model_class) -> None:
    """Check fail when multiple prediction columns are used."""
    # import pytorch losses with multiple predictions capability
    from neuralforecast.losses.pytorch import MQLoss

    # define model
    model = model_class(
        freq="A-DEC",
        loss=MQLoss(quantiles=[0.25, 0.5, 0.75]),
        max_steps=5,
        trainer_kwargs={"logger": False},
    )

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # attempt predict
    with pytest.raises(
        NotImplementedError, match="Multiple prediction columns are not supported."
    ):
        model.predict()


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_auto_freq(model_class) -> None:
    """Test with freq set to 'auto'."""
    # define model
    model = model_class(freq="auto", max_steps=5, trainer_kwargs={"logger": False})

    # train model
    model.fit(y_train, fh=[1, 2, 3, 4])

    # predict with trained model
    y_pred = model.predict()

    # convert freq str to DateOffset object for comparison
    offset_freq = pandas.tseries.frequencies.to_offset(y_train.index.freq)
    offset_auto_freq = pandas.tseries.frequencies.to_offset(y_pred.index.freq)

    assert offset_freq == offset_auto_freq


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.parametrize(
    "freq",
    [
        "B",
        "D",
        "W",
        "M",
        "Q",
        "A",
        "Y",
        "H",
        "T",
        "min",
        "S",
        "L",
        "ms",
        "U",
        "us",
        "N",
    ],
)
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_auto_against_given_freq(model_class, freq) -> None:
    """Test NeuralForecastRNN with freq set to 'auto' on all freqs."""
    # prepare data
    y = pandas.Series(
        data=range(10),
        index=pandas.date_range(start="2024-01-01", periods=10, freq=freq),
    )

    # define model
    model = model_class(freq="auto", max_steps=1, trainer_kwargs={"logger": False})

    # attempt train
    model.fit(y, fh=[1, 2, 3, 4])

    # convert freq str to DateOffset object for comparison
    offset_freq = pandas.tseries.frequencies.to_offset(freq)
    offset_auto_freq = pandas.tseries.frequencies.to_offset(model._freq)

    assert offset_freq == offset_auto_freq


@pytest.mark.parametrize(
    "index, freq",
    [
        # RangeIndex
        (pandas.RangeIndex(start=0, stop=20), 1),
        (pandas.RangeIndex(start=0, stop=20, step=3), 3),
        # Index
        (pandas.Index(range(20)), 1),
        (pandas.Index([1, 4, 7, 10, 13, 16]), 3),
        # DatetimeIndex
        (pandas.date_range(start="2024-01-01", periods=10), "D"),
        (pandas.date_range(start="2024-01-01", periods=10, freq="M"), "M"),
        # PeriodIndex
        (pandas.period_range(start="2024-01-01", periods=10), "D"),
        (pandas.period_range(start="2024-01-01", periods=10, freq="M"), "M"),
        (pandas.period_range(start="2024-01-01", periods=10).drop(["2024-01-02"]), "D"),
    ],
)
@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_auto_freq_on_valid_index(
    index, freq, model_class
) -> None:
    """Test with freq set to 'auto' on valid indexes (equispaced dates)."""
    y = pandas.Series(data=range(len(index)), index=index)

    model = model_class(freq=freq, max_steps=1, trainer_kwargs={"logger": False})
    model_auto = model_class(freq="auto", max_steps=1, trainer_kwargs={"logger": False})

    model.fit(y, fh=[1, 2, 3])
    model_auto.fit(y, fh=[1, 2, 3])

    pred = model.predict()
    pred_auto = model_auto.predict()

    # check prediction
    pandas.testing.assert_series_equal(pred, pred_auto)


@pytest.mark.parametrize(
    "index",
    [
        # RangeIndex is always equispaced
        # Index
        pandas.Index([1, 2, 3, 4, 5, 7])
    ],
)
@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_auto_freq_on_missing_int_like(index, model_class) -> None:
    """Test with freq set to 'auto' on int-like index with missing values."""
    y = pandas.Series(data=range(len(index)), index=index)

    model = model_class(freq="auto", max_steps=1, trainer_kwargs={"logger": False})

    with pytest.raises(
        ValueError,
        match="(could not interpret freq).*(use a valid integer offset in index)",
    ):
        model.fit(y, fh=[1, 2, 3])


@pytest.mark.parametrize(
    "index",
    [
        # PeriodIndex: freq is preserved in index even in missing data
        # DatetimeIndex
        pandas.date_range(start="2024-01-01", periods=5).drop(["2024-01-02"]),
        pandas.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04", "2000-01-05"]),
    ],
)
@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_auto_freq_on_missing_date_like(
    index, model_class
) -> None:
    """Test with freq set to 'auto' on date-like index with missing values."""
    y = pandas.Series(data=range(len(index)), index=index)

    model = model_class(freq="auto", max_steps=1, trainer_kwargs={"logger": False})

    with pytest.raises(
        ValueError, match="(could not interpret freq).*(use a valid offset in index)"
    ):
        model.fit(y, fh=[1, 2, 3])
