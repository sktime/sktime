"""Tests for CINNForecaster."""

__author__ = ["imsurya26"]

import pytest

from sktime.forecasting.cinn import CINNForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.series import _make_series


def _subnet_hidden_widths(forecaster):
    """Return the hidden widths of the coupling-layer subnets in the network.

    Each subnet is ``Sequential(Linear(ch_in, hidden), activation, Linear(...))``,
    so the hidden width is the output size of its first layer.
    """
    from torch import nn

    return {
        m.out_features
        for name, m in forecaster.network.named_modules()
        if isinstance(m, nn.Linear) and ".subnet" in name and name.endswith(".0")
    }


@pytest.mark.skipif(
    not run_test_for_class(CINNForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cinn_hidden_dim_size_is_used():
    """Test that hidden_dim_size sets the subnet width, also after save and load.

    Previously, hidden_dim_size was not passed to the network,
    so the subnets always had the network default width of 64.
    """
    params = CINNForecaster.get_test_params()[0]
    y = _make_series(n_timepoints=60, all_positive=True)

    forecaster = CINNForecaster(hidden_dim_size=8, **params)
    forecaster.fit(y, fh=[1, 2, 3])

    assert _subnet_hidden_widths(forecaster) == {8}

    loaded = CINNForecaster.load_from_serial(forecaster.save()[1])

    assert _subnet_hidden_widths(loaded) == {8}
    loaded.predict(fh=[1, 2, 3])
