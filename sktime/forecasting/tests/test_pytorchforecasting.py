# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for interfacing estimators from pytorch-forecasting."""

import os

import pytest
from sklearn.model_selection import train_test_split

from sktime.datatypes._utilities import get_cutoff
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import (
    PytorchForecastingDeepAR,
    PytorchForecastingNBeats,
    PytorchForecastingNHiTS,
    PytorchForecastingTFT,
)
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import (
    _assert_correct_columns,
    _assert_correct_pred_time_index,
)
from sktime.utils._testing.hierarchical import _make_hierarchical

__author__ = ["XinyuWu"]


@pytest.mark.parametrize(
    "model_class",
    [
        PytorchForecastingDeepAR,
        PytorchForecastingNBeats,
        PytorchForecastingNHiTS,
        PytorchForecastingTFT,
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(
        [
            PytorchForecastingDeepAR,
            PytorchForecastingNBeats,
            PytorchForecastingNHiTS,
            PytorchForecastingTFT,
        ]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_load_model_from_disk(model_class) -> None:
    """Test load fitted model from disk with refit."""
    # define model
    model = model_class(**(model_class.get_test_params()[0]))

    # generate data
    data_length = 100
    data = _make_hierarchical(
        (5, 100),
        n_columns=2,
        max_timepoints=data_length,
        min_timepoints=data_length,
    )
    x = data["c0"].to_frame()
    y = data["c1"].to_frame()
    X_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.1, train_size=0.9, shuffle=False
    )
    _, X_test, _, y_test = train_test_split(
        x, y, test_size=0.2, train_size=0.8, shuffle=False
    )
    max_prediction_length = 3
    fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)

    # fit the model to generate the checkpoint
    model.fit(y_train, X_train, fh=fh)
    # get the best model path
    if model._trainer.checkpoint_callback is not None:
        best_model_path = model._trainer.checkpoint_callback.best_model_path
    else:
        best_model_path = getattr(model, "_random_log_dir", None)
        if best_model_path is None:
            best_model_path = model._gen_random_log_dir()
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        best_model_path = best_model_path + "/last_model.pt"
        model._trainer.save_checkpoint(best_model_path)

    # reload the model from best_model_path
    model = model_class(model_path=best_model_path, **model_class.get_test_params()[0])
    # call fit function (no real fitting will happen)
    model.fit(y_train, X_train, fh=fh)

    # verify the actual fit is skipped by checking the _trainer attribute
    # there should be no _trainer attribute
    try:
        model._trainer
        raise AssertionError("Trainer should not be initialized")
    except AttributeError:  # noqa: S110
        pass

    # remove max_prediction_length from the end of y_test
    len_levels = len(y_test.index.names)
    y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
        lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    )

    # predict with model loaded from disk
    y_pred = model.predict(fh=fh, X=X_test, y=y_test)

    # check prediction index and column names
    cutoff = get_cutoff(y_test, return_index=True)
    index_pred = y_pred.iloc[:max_prediction_length].index.get_level_values(2)
    _assert_correct_pred_time_index(index_pred, cutoff, fh)
    _assert_correct_columns(y_pred, y_test)


def _make_train_test_data(data_length=80):
    """Create hierarchical train/test data for pytorch-forecasting adapter tests."""
    data = _make_hierarchical(
        (2, 20),
        n_columns=2,
        max_timepoints=data_length,
        min_timepoints=data_length,
    )
    x = data["c0"].to_frame()
    y = data["c1"].to_frame()
    X_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.1, train_size=0.9, shuffle=False
    )
    _, X_test, _, y_test = train_test_split(
        x, y, test_size=0.2, train_size=0.8, shuffle=False
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.skipif(
    not run_test_for_class(PytorchForecastingNHiTS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_nhits_uses_model_length_params_when_fh_is_shorter():
    """Test that NHiTS preserves model context/prediction length from model_params."""
    X_train, X_test, y_train, y_test = _make_train_test_data()
    fh_max = 3
    fh = ForecastingHorizon(range(1, fh_max + 1), is_relative=True)

    len_levels = len(y_test.index.names)
    y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
        lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-fh_max]
    )

    model = PytorchForecastingNHiTS(
        model_params={
            "context_length": 10,
            "prediction_length": 6,
            "hidden_size": 4,
            "n_blocks": [1, 1],
            "n_layers": 1,
            "log_interval": -1,
        },
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 2,
            "enable_checkpointing": False,
            "logger": False,
        },
        train_to_dataloader_params={"batch_size": 2},
    )

    model.fit(y_train, X_train, fh=fh)
    y_pred = model.predict(fh=fh, X=X_test, y=y_test)

    assert model._forecaster.hparams.context_length == 10
    assert model._forecaster.hparams.prediction_length == 6
    assert model._forecaster.hparams.hidden_size == 4

    cutoff = get_cutoff(y_test, return_index=True)
    index_pred = y_pred.iloc[:fh_max].index.get_level_values(2)
    _assert_correct_pred_time_index(index_pred, cutoff, fh)
    _assert_correct_columns(y_pred, y_test)


@pytest.mark.skipif(
    not run_test_for_class(PytorchForecastingNHiTS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_nhits_raises_if_fh_exceeds_configured_prediction_length():
    """Test that fh cannot exceed a fixed model prediction length."""
    X_train, _, y_train, _ = _make_train_test_data()
    fh = ForecastingHorizon(range(1, 4), is_relative=True)
    model = PytorchForecastingNHiTS(
        model_params={
            "prediction_length": 2,
            "hidden_size": 4,
            "n_blocks": [1, 1],
            "n_layers": 1,
            "log_interval": -1,
        },
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 1,
            "enable_checkpointing": False,
            "logger": False,
        },
        train_to_dataloader_params={"batch_size": 2},
    )

    with pytest.raises(ValueError, match="maximum step in `fh`.*less than or equal to"):
        model.fit(y_train, X_train, fh=fh)


@pytest.mark.skipif(
    not run_test_for_class(PytorchForecastingNHiTS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_nhits_refit_can_increase_prediction_length_if_not_fixed():
    """Test that inferred prediction_length tracks fh across refits."""
    X_train, _, y_train, _ = _make_train_test_data()

    model = PytorchForecastingNHiTS(
        model_params={
            "hidden_size": 4,
            "n_blocks": [1, 1],
            "n_layers": 1,
            "log_interval": -1,
        },
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 1,
            "enable_checkpointing": False,
            "logger": False,
        },
        train_to_dataloader_params={"batch_size": 2},
    )

    fh_short = ForecastingHorizon(range(1, 3), is_relative=True)
    fh_long = ForecastingHorizon(range(1, 5), is_relative=True)

    model.fit(y_train, X_train, fh=fh_short)
    assert model._forecaster.hparams.prediction_length == 2

    model.fit(y_train, X_train, fh=fh_long)
    assert model._forecaster.hparams.prediction_length == 4
