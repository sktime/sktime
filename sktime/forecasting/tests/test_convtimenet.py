"""Tests for ConvTimeNetForecaster."""

import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip if torch not installed",
)
def test_convtimenet_random_state_zero_sets_seed():
    """Regression test: random_state=0 must set the generator seed.

    Previously, build_pytorch_train_dataloader and the pred dataloader used
    `if self.random_state:` which evaluates random_state=0 as falsy.
    This meant seed 0 was silently ignored and results were not reproducible
    even though the user explicitly passed random_state=0.

    Fix: changed to `if self.random_state is not None:` so seed 0 is applied.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.convtimenet import ConvTimeNetForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    params = ConvTimeNetForecaster.get_test_params()
    if isinstance(params, list):
        params = params[0]
    params["random_state"] = 0
    forecaster = ConvTimeNetForecaster(**params)

    # self.network is only set after fit(), so mock it directly
    forecaster.network = MagicMock()
    forecaster.network.seq_len = params.get("context_window", 10)
    forecaster.network.pred_len = params.get("pred_len", 3)

    with patch("torch.Generator") as mock_generator_class:
        mock_gen = MagicMock()
        mock_generator_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            with patch("sktime.forecasting.base.adapters._pytorch.PyTorchTrainDataset"):
                forecaster.build_pytorch_train_dataloader(y)
    # manual_seed must have been called with 0
    mock_gen.manual_seed.assert_called_once_with(0)
