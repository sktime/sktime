"""Tests for Chronos2Forecaster."""

__author__ = ["amethystani"]

import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class("Chronos2Forecaster"),
    reason="run test only if soft dependencies are present and incrementally",
)
def test_chronos2_context_truncation_uses_time_axis():
    """Regression test for context truncation axis bug.

    _fit was checking context.shape[1] (features/columns) instead of
    context.shape[0] (time steps/rows) when deciding whether to truncate
    the context to context_length. On a series with many time points but
    few variables, shape[1] < context_length so truncation never triggered.
    """
    from unittest.mock import MagicMock, patch

    from sktime.forecasting.chronos2 import Chronos2Forecaster

    n_timepoints = 100
    context_length = 10
    y = pd.DataFrame(
        {
            "a": range(n_timepoints),
            "b": range(100, 100 + n_timepoints),
        }
    )

    mock_pipeline = MagicMock()
    mock_pipeline.model_context_length = context_length

    with patch.object(Chronos2Forecaster, "_load_pipeline", return_value=mock_pipeline):
        f = Chronos2Forecaster(context_length=context_length)
        f._fit(y=y, X=None, fh=None)

    # _context is stored transposed: shape (n_vars, n_timepoints_kept)
    # so shape[1] is the number of time steps retained
    assert f._context.shape[1] == context_length, (
        f"Context should be truncated to {context_length} time steps "
        f"along the time axis; got shape {f._context.shape}"
    )
