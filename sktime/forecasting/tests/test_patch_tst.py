"""Tests for the PatchTST forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.patch_tst import PatchTSTForecaster
from sktime.tests.test_switch import run_test_for_class

# public checkpoint whose native prediction_length is 96, loading with no
# missing or mismatched weights
_PATCHTST_MODEL = "ibm-granite/granite-timeseries-patchtst"
_CONTEXT_LENGTH = 512
_PREDICTION_LENGTH = 96
_N_CHANNELS = 7


def _make_y():
    """Return data matching the checkpoint's expected context and channels."""
    values = np.random.RandomState(0).randn(_CONTEXT_LENGTH, _N_CHANNELS)
    columns = [f"c{i}" for i in range(_N_CHANNELS)]
    return pd.DataFrame(values.astype("float32"), columns=columns)


@pytest.mark.skipif(
    not run_test_for_class(PatchTSTForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_patch_tst_zero_shot_native_prediction_length():
    """Zero-shot forecasting works for the checkpoint's full prediction_length.

    ``_fit`` derives ``prediction_length`` from the forecasting horizon. Requesting
    the checkpoint's own ``prediction_length`` must not enlarge it, as that reshapes
    the prediction head and makes the pretrained weights fail to load.
    """
    y = _make_y()
    fh = list(range(1, _PREDICTION_LENGTH + 1))

    forecaster = PatchTSTForecaster(
        model_path=_PATCHTST_MODEL, fit_strategy="zero-shot"
    )
    y_pred = forecaster.fit(y, fh=fh).predict()

    assert len(y_pred) == _PREDICTION_LENGTH
    assert forecaster.model.config.prediction_length == _PREDICTION_LENGTH
