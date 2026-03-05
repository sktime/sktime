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
def test_pass_initialized_model(model_class) -> None:
    """Test passing a pre-initialized pytorch-forecasting model directly."""
    # get_test_params returns broadcasting variants first, then non-broadcasting.
    # we need non-broadcasting here since the test uses hierarchical (global) data.
    all_params = model_class.get_test_params()
    params = next(p for p in all_params if not p.get("broadcasting", False))

    base_model = model_class(**params)

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

    # train normally first to get a live pytorch-forecasting model object
    base_model.fit(y_train, X_train, fh=fh)
    trained_pf_model = base_model.best_model

    # pass that model object directly into a fresh wrapper — no training should happen
    model = model_class(model=trained_pf_model, **params)
    model.fit(y_train, X_train, fh=fh)

    # no trainer should have been created since we skipped training
    assert not hasattr(model, "_trainer"), (
        "Trainer should not be initialized when a pre-built model is passed"
    )
    assert model.best_model is trained_pf_model

    # trim the last few steps from y_test so predict has something to forecast into
    len_levels = len(y_test.index.names)
    y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
        lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    )

    y_pred = model.predict(fh=fh, X=X_test, y=y_test)

    cutoff = get_cutoff(y_test, return_index=True)
    index_pred = y_pred.iloc[:max_prediction_length].index.get_level_values(2)
    _assert_correct_pred_time_index(index_pred, cutoff, fh)
    _assert_correct_columns(y_pred, y_test)


def test_model_and_model_path_mutually_exclusive() -> None:
    """Test that passing both model and model_path raises a ValueError."""
    with pytest.raises(ValueError, match="Only one of 'model' or 'model_path'"):
        PytorchForecastingNBeats(model=object(), model_path="some/path.ckpt")
