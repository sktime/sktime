# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test MIRA time preparation without loading a model checkpoint."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.mira import _prepare_context
from sktime.utils.dependencies import _check_soft_dependencies

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="MIRA time preparation requires torch",
)


@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("context_length", [None, 12])
@pytest.mark.parametrize("tz", [None, "Asia/Kolkata"])
def test_mira_datetime_spacing(freq, context_length, tz):
    """Adjacent dates remain distinct, including across the forecast boundary."""
    index = pd.date_range("2026-01-01", periods=48, freq=freq, tz=tz)
    y = pd.DataFrame({"y": np.arange(48.0)}, index=index)
    values, times = _prepare_context(y, 4, index[-1:], context_length)

    n_context = 48 if context_length is None else context_length
    np.testing.assert_array_equal(values.numpy()[0], y.iloc[-n_context:, 0])
    assert times.shape == (1, n_context + 4)
    assert np.unique(times.numpy()).size == n_context + 4
    # CT-RoPE rescales the full context and future interval to [0, seq_len - 1].
    normalized = (times - times[:, :1]) / (times[:, -1:] - times[:, :1])
    normalized = normalized * (times.shape[1] - 1)
    np.testing.assert_allclose(
        normalized.numpy()[0], np.arange(n_context + 4), atol=1e-5
    )


def test_mira_irregular_datetime_spacing():
    """Rebasing preserves irregular elapsed durations rather than row positions."""
    regular = pd.date_range("2026-01-01", periods=10, freq="s")
    index = regular[[0, 1, 4, 9]]
    y = pd.DataFrame({"y": np.arange(4.0)}, index=index)
    _, times = _prepare_context(y, 2, regular[-1:], None)
    np.testing.assert_allclose(
        times.numpy()[0] / times.numpy()[0, 1], [0, 1, 4, 9, 10, 11], atol=1e-5
    )


@pytest.mark.parametrize(
    "index",
    [pd.period_range("2026-01", periods=48, freq="M"), pd.RangeIndex(48)],
)
def test_mira_other_index_spacing(index):
    """Period and integer indices retain their existing unit spacing."""
    y = pd.DataFrame({"y": np.arange(48.0)}, index=index)
    _, times = _prepare_context(y, 4, index[-1:], None)
    np.testing.assert_array_equal(np.diff(times.numpy()[0]), np.ones(51))
