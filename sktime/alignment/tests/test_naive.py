# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for AlignerNaive."""

import numpy as np
import pandas as pd
import pytest

from sktime.alignment.naive import AlignerNaive
from sktime.tests.test_switch import run_test_for_class


def _make_series(n=10):
    return [
        pd.DataFrame({"a": np.arange(n)}),
        pd.DataFrame({"a": np.arange(n + 3)}),
    ]


@pytest.mark.skipif(
    not run_test_for_class(AlignerNaive),
    reason="run_test_for_class returned False, skipping per testing policy",
)
def test_aligner_naive_invalid_strategy_error_message():
    """Invalid strategy raises ValueError mentioning the bad value and valid options."""
    aligner = AlignerNaive(strategy="not-a-valid-strategy")
    with pytest.raises(ValueError) as exc_info:
        aligner.fit(_make_series())
    msg = str(exc_info.value)
    assert "not-a-valid-strategy" in msg
    assert "start" in msg
    assert "end" in msg
    assert "start-end" in msg


@pytest.mark.skipif(
    not run_test_for_class(AlignerNaive),
    reason="run_test_for_class returned False, skipping per testing policy",
)
@pytest.mark.parametrize("strategy", ["start", "end", "start-end"])
def test_aligner_naive_valid_strategies(strategy):
    """Valid strategies fit without raising."""
    aligner = AlignerNaive(strategy=strategy)
    aligner.fit(_make_series())
