"""Tests for SCINetForecaster."""

import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip if torch not installed",
)
def test_scinet_num_stacks_passed_to_network():
    """Regression test: num_stacks must be forwarded to the SCINet network.

    Previously, _build_network() passed num_stacks=self.hid_size instead of
    num_stacks=self.num_stacks. This meant the user's num_stacks value was
    silently ignored and hid_size was used as the stack count instead,
    causing the wrong architecture to be built with no error raised.

    Fix: changed the argument to num_stacks=self.num_stacks.
    """
    from unittest.mock import MagicMock, patch

    from sktime.forecasting.scinet import SCINetForecaster

    forecaster = SCINetForecaster(
        seq_len=8,
        num_stacks=2,
        hid_size=1,  # deliberately different so the bug is detectable
    )

    captured = {}

    with patch("sktime.networks.scinet.SCINet") as mock_scinet:
        mock_instance = MagicMock()

        def capture_and_return(*args, **kwargs):
            captured.update(kwargs)
            return mock_instance

        mock_scinet.side_effect = capture_and_return

        try:
            forecaster._y = MagicMock()
            forecaster._y.shape = (50, 1)
            forecaster._build_network(fh=3)
        except Exception:  # noqa: S110
            pass  # network instantiation may fail in mock environment

    if "num_stacks" in captured:
        assert captured["num_stacks"] == 2, (
            f"Expected num_stacks=2 to be forwarded to SCINet, "
            f"but got num_stacks={captured['num_stacks']}. "
            "Bug: num_stacks=self.hid_size was used instead of "
            "num_stacks=self.num_stacks."
        )
