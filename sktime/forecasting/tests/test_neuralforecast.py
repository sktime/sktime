# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for NeuralForecastRNN."""
import pandas
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.neuralforecast import NeuralForecastRNN
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

__author__ = ["yarnabrina"]

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)


@pytest.mark.skipif(
    not run_test_for_class(NeuralForecastRNN),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_rnn_univariate_y_without_X() -> None:
    """Test NeuralForecastRNN with single endogenous without exogenous."""
    # define model
    model = NeuralForecastRNN("A-DEC", max_steps=5, trainer_kwargs={"logger": False})

    # attempt fit with negative fh
    with pytest.raises(
        NotImplementedError, match="in-sample prediction is currently not supported"
    ):
        model.fit(y_train, fh=[-2, -1, 0, 1, 2])

    # train model
    model.fit(y_train, fh=[1, 2, 3, 4])

    # predict with trained model
    y_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(y_pred.index, y_test.index, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(NeuralForecastRNN),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_rnn_univariate_y_with_X() -> None:
    """Test NeuralForecastRNN with single endogenous with exogenous."""
    # select feature columns
    exog_list = ["GNPDEFL", "GNP", "UNEMP"]

    # define model
    model = NeuralForecastRNN(
        "A-DEC", futr_exog_list=exog_list, max_steps=5, trainer_kwargs={"logger": False}
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


@pytest.mark.skipif(
    not run_test_for_class(NeuralForecastRNN),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_rnn_multivariate_y_without_X() -> None:
    """Test NeuralForecastRNN with multiple endogenous without exogenous."""
    # define model
    model = NeuralForecastRNN("A-DEC", max_steps=5, trainer_kwargs={"logger": False})

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # predict with trained model
    X_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(X_pred.index, X_test.index, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(NeuralForecastRNN),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_rnn_with_non_default_loss() -> None:
    """Test NeuralForecastRNN with multiple endogenous without exogenous."""
    # import non-default pytorch losses
    from neuralforecast.losses.pytorch import MASE, HuberQLoss

    # define model
    model = NeuralForecastRNN(
        "A-DEC",
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


@pytest.mark.skipif(
    not run_test_for_class(NeuralForecastRNN),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_rnn_fail_with_multiple_predictions() -> None:
    """Check NeuralForecastRNN fail when multiple prediction columns are used."""
    # import pytorch losses with multiple predictions capability
    from neuralforecast.losses.pytorch import MQLoss

    # define model
    model = NeuralForecastRNN(
        "A-DEC",
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
