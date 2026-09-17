"""Tests for PatchTSMixerForecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.patch_tsmixer import PatchTSMixerForecaster

_MODEL_PATH = "ibm-granite/granite-timeseries-patchtsmixer"
_MODEL_REVISION = "90dc5a88d45f032b7dceefb5d814ca2af54f2ff9"
_TARGET_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@pytest.mark.skipif(
    not _check_estimator_deps(PatchTSMixerForecaster, severity="none"),
    reason="PatchTSMixerForecaster soft dependencies not available",
)
def test_patch_tsmixer_predictions_match_source_reference():
    """Predictions match the original IBM/Hugging Face implementation."""
    time = np.arange(512, dtype=np.float32)
    values = {
        name: (offset + np.sin(time / (8.0 + offset)) + 0.002 * time).astype(np.float32)
        for offset, name in enumerate(_TARGET_COLUMNS, start=1)
    }
    y = pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=512, freq="h"))
    forecaster = PatchTSMixerForecaster(
        model_path=_MODEL_PATH,
        revision=_MODEL_REVISION,
        context_length=512,
        prediction_length=96,
        validation_split=0.0,
        train_model=False,
        scaling=False,
    )

    y_pred = forecaster.fit(y, fh=range(1, 97)).predict(fh=range(1, 97))

    expected = np.asarray(
        [
            [
                2.46599293,
                2.63427544,
                3.11994505,
                4.97724009,
                5.2243247,
                5.28185463,
                8.13028336,
            ],
            [
                2.30446863,
                2.62986088,
                3.00171471,
                4.88996744,
                5.05688524,
                5.28941774,
                7.91808081,
            ],
            [
                2.19133615,
                2.69998837,
                2.96796489,
                4.89432335,
                5.0068574,
                5.408319,
                7.85999775,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
