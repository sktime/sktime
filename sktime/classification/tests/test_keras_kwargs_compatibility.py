"""Unit tests for sktime TF-classifier compatibility with keras kwargs."""

__author__ = ["achieveordie"]
__all__ = [
    "test_custom_compile_kwargs",
    "test_custom_fit_kwargs",
    "test_custom_compile_and_fit_kwargs",
    "test_fit_kwargs_functional",
]

import pytest

from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.deep_learning.base import (
    KerasCompileKwargs,
    KerasFitKwargs,
)
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_dl_dependencies


@pytest.mark.skipif(
    not _check_dl_dependencies(severity="none")
    or not run_test_module_changed("sktime.classification.deep_learning"),
    reason="skip test if tensorflow not installed or no changes to the DL module",
)
def test_custom_compile_kwargs():
    """Test that custom compile kwargs are stored correctly in classifier."""
    custom_compile_kwargs = KerasCompileKwargs(
        steps_per_execution=2,
        auto_scale_loss=False,
    )
    classifier = CNNClassifier(n_epochs=1, compile_kwargs=custom_compile_kwargs)
    assert (
        classifier.compile_kwargs is not None
        and not classifier.compile_kwargs.auto_scale_loss
        and classifier.compile_kwargs.steps_per_execution == 2
    )


@pytest.mark.skipif(
    not _check_dl_dependencies(severity="none")
    or not run_test_module_changed("sktime.classification.deep_learning"),
    reason="skip test if tensorflow not installed or no changes to the DL module",
)
def test_custom_fit_kwargs():
    """Test that custom fit kwargs are stored correctly in classifier."""
    custom_fit_kwargs = KerasFitKwargs(shuffle=False, steps_per_epoch=2)
    classifier = CNNClassifier(n_epochs=1, fit_kwargs=custom_fit_kwargs)
    assert (
        classifier.fit_kwargs is not None
        and not classifier.fit_kwargs.shuffle
        and classifier.fit_kwargs.steps_per_epoch == 2
    )


@pytest.mark.skipif(
    not _check_dl_dependencies(severity="none")
    or not run_test_module_changed("sktime.classification.deep_learning"),
    reason="skip test if tensorflow not installed or no changes to the DL module",
)
def test_custom_compile_and_fit_kwargs():
    """Test that both compile and fit kwargs can be used together."""
    custom_compile_kwargs = KerasCompileKwargs(
        steps_per_execution=2,
        auto_scale_loss=False,
    )
    custom_fit_kwargs = KerasFitKwargs(shuffle=False, steps_per_epoch=2)
    classifier = CNNClassifier(
        n_epochs=1,
        compile_kwargs=custom_compile_kwargs,
        fit_kwargs=custom_fit_kwargs,
    )

    assert (
        classifier.compile_kwargs is not None
        and not classifier.compile_kwargs.auto_scale_loss
        and classifier.compile_kwargs.steps_per_execution == 2
    )
    assert (
        classifier.fit_kwargs is not None
        and not classifier.fit_kwargs.shuffle
        and classifier.fit_kwargs.steps_per_epoch == 2
    )


@pytest.mark.skipif(
    not _check_dl_dependencies(severity="none")
    or not run_test_module_changed("sktime.classification.deep_learning"),
    reason="skip test if tensorflow not installed or no changes to the DL module",
)
def test_fit_kwargs_functional():
    """Test that fit_kwargs actually affect training behavior.

    This test verifies that validation_split produces validation metrics
    in the training history.
    """
    X_train, y_train = load_unit_test(split="train")

    # Train with validation_split to generate validation metrics
    fit_kwargs = KerasFitKwargs(validation_split=0.2)
    classifier = CNNClassifier(
        n_epochs=2,
        batch_size=4,
        fit_kwargs=fit_kwargs,
    )

    classifier.fit(X_train, y_train)

    # Verify that validation metrics were computed
    history = classifier.summary()
    assert history is not None, "Training history should not be None"
    assert "val_loss" in history, "Validation loss should be in history"
    assert "val_accuracy" in history, "Validation accuracy should be in history"
    assert len(history["val_loss"]) == 2, "Should have validation loss for 2 epochs"
