"""Tests for the MIRAForecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.mira import MIRAForecaster

pytestmark = pytest.mark.skipif(
    not _check_estimator_deps(MIRAForecaster, severity="none"),
    reason="skip test if required soft dependencies are not available",
)


@pytest.fixture
def mira_forecaster(monkeypatch):
    """MIRAForecaster with the model loading stubbed out.

    ``_load_model`` is monkeypatched so the test does not download the MIRA
    checkpoint. ``_fit`` only calls ``self.model.eval()`` on the returned
    object, so a plain mock is sufficient.
    """
    pytest.importorskip("torch")

    forecaster = MIRAForecaster(context_length=4)
    mock_model = type("MockModel", (), {"eval": lambda self: None})()
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_model)
    return forecaster


def test_mira_update_restricts_context_to_context_length(mira_forecaster):
    """``_update`` retains only the latest ``context_length`` observations.

    MIRA is zero-shot: prediction only needs the trailing ``context_length``
    points (see ``_prepare_context``). After ``update``, ``self._y`` must be
    trimmed to ``context_length`` rather than accumulating all history.
    """
    y = pd.DataFrame({"y": np.arange(20, dtype=np.float64)})
    mira_forecaster.fit(y, fh=1)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=np.float64)}, index=pd.RangeIndex(20, 25)
    )
    mira_forecaster.update(y_new, update_params=False)

    assert len(mira_forecaster._y) == 4
    # the retained window is the tail of the combined series
    np.testing.assert_array_equal(
        mira_forecaster._y["y"].to_numpy(), [21, 22, 23, 24]
    )


def test_mira_update_no_context_length_keeps_all_data(monkeypatch):
    """With ``context_length=None`` all history is retained (MIRA uses all)."""
    pytest.importorskip("torch")

    forecaster = MIRAForecaster(context_length=None)
    mock_model = type("MockModel", (), {"eval": lambda self: None})()
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_model)

    y = pd.DataFrame({"y": np.arange(20, dtype=np.float64)})
    forecaster.fit(y, fh=1)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=np.float64)}, index=pd.RangeIndex(20, 25)
    )
    forecaster.update(y_new, update_params=False)

    assert len(forecaster._y) == 25
