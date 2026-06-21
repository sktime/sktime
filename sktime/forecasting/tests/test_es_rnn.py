"""Tests for ESRNNForecaster."""

import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip if torch not installed",
)
def test_esrnn_pred_dataloader_uses_pred_dataset_not_train():
    """Regression test: build_pytorch_pred_dataloader must use custom_dataset_pred.

    Previously, build_pytorch_pred_dataloader incorrectly called
    custom_dataset_train.build_dataset() and set dataset = custom_dataset_train
    when a user passed custom_dataset_pred. This meant the custom prediction
    dataset was silently ignored and the training dataset was used instead,
    producing wrong predictions with no error raised.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.es_rnn import ESRNNForecaster

    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2000-01", periods=20, freq="M"),
    )

    mock_train = MagicMock()
    mock_pred = MagicMock()

    forecaster = ESRNNForecaster(
        custom_dataset_train=mock_train,
        custom_dataset_pred=mock_pred,
    )

    with patch("torch.utils.data.DataLoader") as mock_dataloader:
        forecaster.build_pytorch_pred_dataloader(y, fh=[1, 2, 3])

    mock_pred.build_dataset.assert_called_once_with(y)
    mock_train.build_dataset.assert_not_called()

    mock_dataloader.assert_called_once()
    first_arg = mock_dataloader.call_args[0][0]
    assert first_arg is mock_pred, (
        "DataLoader was given custom_dataset_train instead of "
        "custom_dataset_pred - copy-paste bug in pred dataloader"
    )
