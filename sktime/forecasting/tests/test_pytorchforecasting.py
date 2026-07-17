# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for interfacing estimators from pytorch-forecasting."""

import os

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
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

__author__ = ["XinyuWu", "Nischal1425"]


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
def test_predict_with_future_exogenous_x(model_class):
    """Test predict with future-only exogenous X on non-hierarchical data.

    Verifies that passing only future X rows at predict time (no overlap with
    training X) works correctly. Training X is prepended in _Xy_precheck so
    the internal LEFT join in _Xy_to_dataset retains full history.

    Parameters
    ----------
    model_class : class
        One of the four PytorchForecasting model classes.
    """
    from sktime.utils._testing.series import _make_series

    RAND_SEED = 42
    n_train = 20

    y_series = _make_series(n_timepoints=n_train, random_state=RAND_SEED)
    y = pd.DataFrame(y_series, columns=["foo"])

    long_x = _make_series(n_columns=2, n_timepoints=n_train + 1, random_state=RAND_SEED)
    X_train = long_x.iloc[:n_train]
    X_test = long_x.iloc[n_train:]

    params = model_class.get_test_params()[0]
    forecaster = model_class(**params)
    forecaster.fit(y=y, X=X_train, fh=1)

    y_pred = forecaster.predict(X=X_test)
    assert y_pred is not None
    assert len(y_pred) == 1


@pytest.mark.skipif(
    not _check_soft_dependencies("pytorch-forecasting", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_xy_precheck_global_mode_does_not_prepend_train_x():
    """Regression: deprecated global predict(y=...) must not prepend train X.

    After #10383, local predict prepends fit-time X to future-only X. Global
    evaluate still calls predict(y=y_hist, X=X_hist_and_future) via the
    deprecation mixin; prepending the global training panel there mixes
    instance sets and yields empty/invalid results.
    """
    import numpy as np

    idx_train = pd.MultiIndex.from_product(
        [["train"], [0, 1], pd.RangeIndex(4)], names=["panel", "inst", "time"]
    )
    idx_pred = pd.MultiIndex.from_product(
        [["test"], [0], pd.RangeIndex(5)], names=["panel", "inst", "time"]
    )
    y_train = pd.DataFrame({"y": np.ones(len(idx_train))}, index=idx_train)
    X_train = pd.DataFrame({"x": np.ones(len(idx_train))}, index=idx_train)
    y_hist = pd.DataFrame({"y": np.ones(4)}, index=idx_pred[:4])
    X_pred = pd.DataFrame({"x": np.ones(len(idx_pred))}, index=idx_pred)
    X_future = pd.DataFrame({"x": np.ones(1)}, index=idx_pred[-1:])

    forecaster = PytorchForecastingDeepAR()
    forecaster._y = y_train
    forecaster._X = X_train
    forecaster._state = "fitted"

    # local path: future-only X is prepended with training X
    X_local, y_local = forecaster._Xy_precheck(X_future)
    assert len(X_local) == len(X_train) + len(X_future)
    assert y_local.equals(y_train)

    # global path (deprecated predict(y=...)): X already has history+future
    with forecaster._temporary_y_swap(X_pred, y_hist):
        assert forecaster._global_forecasting is True
        X_global, _ = forecaster._Xy_precheck(X_pred)
        assert len(X_global) == len(X_pred)
    assert getattr(forecaster, "_global_forecasting", False) is False
