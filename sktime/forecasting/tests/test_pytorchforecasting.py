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

    # verify the actual fit is skiped by checking the _trainer attribute
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
        PytorchForecastingNBeats,
        PytorchForecastingTFT,
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(
        [
            PytorchForecastingNBeats,
            PytorchForecastingTFT,
        ]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_refit_functionality(model_class) -> None:
    """Test refit functionality for incremental training."""

    # Define model with minimal parameters for quick testing
    model = model_class(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 3,
            "limit_val_batches": 3,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
        model_params={"num_blocks": [2, 2]}
        if model_class == PytorchForecastingNBeats
        else {},
        dataset_params={
            "min_encoder_length": 20,
            "max_encoder_length": 20,
        }
        if model_class == PytorchForecastingNBeats
        else {},
    )

    # Generate test data
    data = _make_hierarchical(
        (3, 20), n_columns=2, max_timepoints=100, min_timepoints=100
    )
    x = data["c0"].to_frame()
    y = data["c1"].to_frame()

    # Split data for initial training and refit
    y_initial = y.iloc[:80]
    y_new = y.iloc[80:90]
    X_initial = x.iloc[:80] if model_class == PytorchForecastingTFT else None
    X_new = x.iloc[80:90] if model_class == PytorchForecastingTFT else None

    max_prediction_length = 5
    fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)

    # Initial fit
    model.fit(y_initial, X_initial, fh=fh)

    # Get initial prediction
    y_pred_initial = model.predict(fh)

    # Refit with new data
    model.refit(y=y_new, X=X_new, fh=fh)

    # Get prediction after refit
    y_pred_refit = model.predict(fh)

    # Verify predictions are different (model has been updated)
    assert not y_pred_initial.equals(y_pred_refit)

    # Verify shapes are correct
    expected_length = len(fh) * len(y_new.index.get_level_values(0).unique())
    assert y_pred_refit.shape[0] == expected_length


@pytest.mark.parametrize(
    "model_class",
    [
        PytorchForecastingNBeats,
        PytorchForecastingTFT,
    ],
)
@pytest.mark.skipif(
    not run_test_for_class(
        [
            PytorchForecastingNBeats,
            PytorchForecastingTFT,
        ]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_checkpoint_save_load(model_class) -> None:
    """Test checkpoint save and load functionality."""
    import tempfile

    import numpy as np

    # Define model with minimal parameters for quick testing
    model = model_class(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 3,
            "limit_val_batches": 3,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
        model_params={"num_blocks": [2, 2]}
        if model_class == PytorchForecastingNBeats
        else {},
        dataset_params={
            "min_encoder_length": 20,
            "max_encoder_length": 20,
        }
        if model_class == PytorchForecastingNBeats
        else {},
    )

    # Generate test data
    data = _make_hierarchical(
        (3, 20), n_columns=2, max_timepoints=100, min_timepoints=100
    )
    x = data["c0"].to_frame()
    y = data["c1"].to_frame()

    y_train = y.iloc[:80]
    X_train = x.iloc[:80] if model_class == PytorchForecastingTFT else None

    max_prediction_length = 5
    fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)

    # Fit model
    model.fit(y_train, X_train, fh=fh)
    y_pred_original = model.predict(fh)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")

        # Save checkpoint
        model.save_checkpoint(checkpoint_path)

        # Verify file was created
        assert os.path.exists(checkpoint_path)
        assert os.path.getsize(checkpoint_path) > 0

        # Create new model and load checkpoint
        new_model = model_class(
            trainer_params={
                "max_epochs": 1,
                "limit_train_batches": 3,
                "limit_val_batches": 3,
                "enable_checkpointing": False,
                "logger": False,
                "enable_progress_bar": False,
            },
            model_params={"num_blocks": [2, 2]}
            if model_class == PytorchForecastingNBeats
            else {},
            dataset_params={
                "min_encoder_length": 20,
                "max_encoder_length": 20,
            }
            if model_class == PytorchForecastingNBeats
            else {},
        )

        new_model.load_checkpoint(checkpoint_path)

        # Verify checkpoint data was loaded
        assert hasattr(new_model, "_is_checkpoint_loaded")
        assert new_model._is_checkpoint_loaded

        # Restore model by fitting (this triggers _restore_model_from_checkpoint)
        new_model.fit(y_train, X_train, fh=fh)
        y_pred_loaded = new_model.predict(fh)

        # Predictions should be similar (loaded from checkpoint)
        np.testing.assert_allclose(
            y_pred_original.values, y_pred_loaded.values, rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(
    not run_test_for_class([PytorchForecastingNBeats]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_refit_without_initial_fit_raises_error() -> None:
    """Test that refit raises error when called before initial fit."""
    model = PytorchForecastingNBeats(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 3,
            "limit_val_batches": 3,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
        model_params={"num_blocks": [2, 2]},
        dataset_params={
            "min_encoder_length": 20,
            "max_encoder_length": 20,
        },
    )

    # Generate test data
    data = _make_hierarchical(
        (3, 20), n_columns=1, max_timepoints=100, min_timepoints=100
    )
    y = data["c0"].to_frame().iloc[80:90]
    fh = ForecastingHorizon(range(1, 6), is_relative=True)

    with pytest.raises(ValueError, match="Model must be fitted before calling refit"):
        model.refit(y=y, fh=fh)


@pytest.mark.skipif(
    not run_test_for_class([PytorchForecastingNBeats]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_save_checkpoint_without_fit_raises_error() -> None:
    """Test that save_checkpoint raises error when called before fit."""
    import tempfile

    model = PytorchForecastingNBeats(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 3,
            "limit_val_batches": 3,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
        model_params={"num_blocks": [2, 2]},
        dataset_params={
            "min_encoder_length": 20,
            "max_encoder_length": 20,
        },
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        with pytest.raises(
            ValueError, match="Model must be fitted before saving checkpoint"
        ):
            model.save_checkpoint(checkpoint_path)


@pytest.mark.skipif(
    not run_test_for_class([PytorchForecastingNBeats]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_incremental_training_workflow() -> None:
    """Test the complete incremental training workflow."""
    import tempfile

    model = PytorchForecastingNBeats(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 3,
            "limit_val_batches": 3,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
        model_params={"num_blocks": [2, 2]},
        dataset_params={
            "min_encoder_length": 20,
            "max_encoder_length": 20,
        },
    )

    # Generate test data
    data = _make_hierarchical(
        (3, 20), n_columns=1, max_timepoints=100, min_timepoints=100
    )
    y = data["c0"].to_frame()

    y_initial = y.iloc[:80]
    y_new = y.iloc[80:90]

    fh = ForecastingHorizon(range(1, 6), is_relative=True)

    # Initial training
    model.fit(y=y_initial, fh=fh)

    # Simulate saving checkpoint during training
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "epoch_checkpoint.pth")
        model.save_checkpoint(checkpoint_path)

        # Simulate loading and continuing training (like after interruption)
        resumed_model = PytorchForecastingNBeats(
            trainer_params={
                "max_epochs": 1,
                "limit_train_batches": 3,
                "limit_val_batches": 3,
                "enable_checkpointing": False,
                "logger": False,
                "enable_progress_bar": False,
            },
            model_params={"num_blocks": [2, 2]},
            dataset_params={
                "min_encoder_length": 20,
                "max_encoder_length": 20,
            },
        )

        # Load checkpoint
        resumed_model.load_checkpoint(checkpoint_path)

        # Continue training with new batch of data
        resumed_model.fit(y=y_initial, fh=fh)  # Restore model state
        resumed_model.refit(y=y_new, fh=fh)  # Continue with new data

        # Verify model can make predictions
        y_pred = resumed_model.predict(fh)
        assert y_pred is not None
        assert len(y_pred) > 0
