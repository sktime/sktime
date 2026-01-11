# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for iTransformer forecaster."""

__author__ = ["TenFinges"]

from sktime.datasets import load_airline
from sktime.forecasting.itransformer import ITransformerForecaster
from sktime.utils.dependencies import _check_soft_dependencies


def test_itransformer_init():
    """Test initialization."""
    if not _check_soft_dependencies("torch", severity="none"):
        return

    forecaster = ITransformerForecaster(
        context_length=10, prediction_length=2, d_model=16
    )
    assert forecaster.context_length == 10
    assert forecaster.d_model == 16


def test_itransformer_fit_predict():
    """Test fit and predict."""
    if not _check_soft_dependencies("torch", severity="none"):
        return

    y = load_airline()
    # Use small params for speed
    forecaster = ITransformerForecaster(
        context_length=12,
        prediction_length=4,
        num_epochs=1,
        batch_size=2,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
    )

    forecaster.fit(y, fh=[1, 2, 3, 4])
    y_pred = forecaster.predict()

    assert len(y_pred) == 4
    assert not y_pred.isna().any()


def test_check_estimator_compatibility():
    """Test compatibility with check_estimator."""
    from sktime.utils.estimator_checks import check_estimator

    if not _check_soft_dependencies("torch", severity="none"):
        return

    # Just checking instantiation and basic compliance
    # We filter warnings because PyTorch might warn
    # Pass class to check_estimator to verify get_test_params
    check_estimator(ITransformerForecaster, raise_exceptions=True)
