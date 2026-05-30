"""Tests for random_state support in BaseDeepNetworkPyTorch and subclasses.

Regression tests verifying that:
- All 8 affected forecasters accept random_state and store it correctly
- build_pytorch_train_dataloader seeds torch.Generator when random_state is set
- ESRNNForecaster's own overridden dataloader also seeds the generator
- LTSFTransformerForecaster's own overridden dataloader also seeds the generator
- random_state=0 is correctly treated as a valid seed (not falsy)
- random_state=None does not call manual_seed at all
"""

import pytest

from sktime.utils.dependencies import _check_soft_dependencies

_torch_not_installed = not _check_soft_dependencies("torch", severity="none")


# ---------------------------------------------------------------------------
# Parametrized: verify all 8 forecasters accept random_state in __init__
# ---------------------------------------------------------------------------


def _cinn_f_statistic(x, params):
    """Minimal f_statistic for CINNForecaster tests."""
    return x


_FORECASTER_PARAMS = [
    (
        "sktime.forecasting.scinet",
        "SCINetForecaster",
        {"seq_len": 8, "pred_len": 3},
    ),
    (
        "sktime.forecasting.es_rnn",
        "ESRNNForecaster",
        {"window": 3, "pred_len": 3, "num_epochs": 1},
    ),
    (
        "sktime.forecasting.ltsf",
        "LTSFLinearForecaster",
        {"seq_len": 8, "pred_len": 3, "num_epochs": 1},
    ),
    (
        "sktime.forecasting.ltsf",
        "LTSFDLinearForecaster",
        {"seq_len": 8, "pred_len": 3, "num_epochs": 1},
    ),
    (
        "sktime.forecasting.ltsf",
        "LTSFNLinearForecaster",
        {"seq_len": 8, "pred_len": 3, "num_epochs": 1},
    ),
    (
        "sktime.forecasting.ltsf",
        "LTSFTransformerForecaster",
        {"seq_len": 8, "pred_len": 3, "num_epochs": 1, "context_len": 8},
    ),
    (
        "sktime.forecasting.rbf_forecaster",
        "RBFForecaster",
        {"window_length": 5, "epochs": 1},
    ),
    (
        "sktime.forecasting.conditional_invertible_neural_network",
        "CINNForecaster",
        {
            "num_epochs": 1,
            "window_size": 2,
            "sample_dim": 5,
            "f_statistic": _cinn_f_statistic,
            "init_param_f_statistic": [1, 1],
        },
    ),
]


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
@pytest.mark.parametrize("module,cls_name,params", _FORECASTER_PARAMS)
def test_forecaster_accepts_random_state(module, cls_name, params):
    """All 8 forecasters must accept random_state and store it on self.

    Verifies that random_state=None was added to each forecaster's __init__
    signature and that self.random_state is set correctly. Without this,
    passing random_state would raise TypeError: unexpected keyword argument.
    """
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    forecaster = cls(**params, random_state=42)
    assert forecaster.random_state == 42, (
        f"{cls_name}.random_state should be 42 but got {forecaster.random_state}"
    )

    forecaster_none = cls(**params, random_state=None)
    assert forecaster_none.random_state is None, (
        f"{cls_name}.random_state should be None but got {forecaster_none.random_state}"
    )


# ---------------------------------------------------------------------------
# Base class dataloader seeding (SCINetForecaster uses base class directly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
def test_base_train_dataloader_seeds_generator():
    """Regression test: build_pytorch_train_dataloader in base class seeds generator.

    BaseDeepNetworkPyTorch.build_pytorch_train_dataloader must pass a seeded
    torch.Generator to DataLoader when random_state is set. Without this, the
    DataLoader shuffle order is different every run producing non-reproducible results.

    SCINetForecaster uses the base class dataloader without overriding it,
    so it is used here as the representative test case for the base class path.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.scinet import SCINetForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    forecaster = SCINetForecaster(seq_len=8, random_state=42)

    with patch("torch.Generator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            forecaster.network = MagicMock()
            forecaster.network.seq_len = 8
            forecaster.network.pred_len = 3
            forecaster.build_pytorch_train_dataloader(y)

    mock_gen.manual_seed.assert_called_once_with(42)


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
def test_base_train_dataloader_random_state_zero_seeds_generator():
    """Regression test: random_state=0 must seed the generator.

    0 is a valid seed value. The guard must use ``if self.random_state is not None``
    not ``if self.random_state`` because 0 evaluates as falsy in Python and would
    be silently skipped by the latter form.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.scinet import SCINetForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    forecaster = SCINetForecaster(seq_len=8, random_state=0)

    with patch("torch.Generator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            forecaster.network = MagicMock()
            forecaster.network.seq_len = 8
            forecaster.network.pred_len = 3
            forecaster.build_pytorch_train_dataloader(y)

    mock_gen.manual_seed.assert_called_once_with(0)


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
def test_base_train_dataloader_none_does_not_seed():
    """Regression test: random_state=None must NOT call manual_seed.

    When no seed is requested, the generator must be left unseeded so that
    PyTorch's default random behaviour is preserved.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.scinet import SCINetForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    forecaster = SCINetForecaster(seq_len=8, random_state=None)

    with patch("torch.Generator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            forecaster.network = MagicMock()
            forecaster.network.seq_len = 8
            forecaster.network.pred_len = 3
            forecaster.build_pytorch_train_dataloader(y)

    mock_gen.manual_seed.assert_not_called()


# ---------------------------------------------------------------------------
# ESRNN: has its own overridden build_pytorch_train_dataloader
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
def test_esrnn_train_dataloader_seeds_generator():
    """Regression test: ESRNNForecaster's overridden dataloader seeds generator.

    ESRNNForecaster overrides build_pytorch_train_dataloader with its own
    implementation using ESRNNTrainDataset. The base class fix alone is not
    enough — this override must also seed the torch.Generator.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.es_rnn import ESRNNForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    forecaster = ESRNNForecaster(
        window=5,
        pred_len=3,
        num_epochs=1,
        random_state=42,
    )

    with patch("torch.Generator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            forecaster.network = MagicMock()
            forecaster.network.pred_len = 3
            forecaster.build_pytorch_train_dataloader(y)

    mock_gen.manual_seed.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# LTSFTransformer: has its own overridden build_pytorch_train_dataloader
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_torch_not_installed, reason="skip if torch not installed")
def test_ltsf_transformer_train_dataloader_seeds_generator():
    """Regression test: LTSFTransformerForecaster overridden dataloader seed generator.

    LTSFTransformerForecaster overrides build_pytorch_train_dataloader with its
    own implementation using PytorchFormerDataset. The base class fix alone is
    not enough — this override must also seed the torch.Generator.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from sktime.forecasting.ltsf import LTSFTransformerForecaster

    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2000-01", periods=50, freq="ME"),
    )

    forecaster = LTSFTransformerForecaster(
        seq_len=8,
        context_len=8,
        pred_len=3,
        num_epochs=1,
        random_state=42,
    )

    # Set internal state normally written during fit
    forecaster._pred_len = 3
    forecaster._temporal_encoding = False
    forecaster.temporal_encoding_type = "linear"
    forecaster.freq = "M"

    with patch("torch.Generator") as mock_gen_class:
        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen

        with patch("torch.utils.data.DataLoader"):
            with patch("sktime.networks.ltsf.data.dataset.PytorchFormerDataset"):
                forecaster.build_pytorch_train_dataloader(y)

    mock_gen.manual_seed.assert_called_once_with(42)
