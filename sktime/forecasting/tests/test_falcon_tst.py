"""Golden-output regression tests for the FalconTST forecaster."""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.falcon_tst import FalconTSTForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated using the original Hugging Face implementation
# (ant-intl/Falcon-TST_Large) directly on load_airline:
#
#   import numpy as np, torch
#   from transformers import AutoModel
#   from sktime.datasets import load_airline
#
#   y = load_airline()
#   model = AutoModel.from_pretrained(
#       "ant-intl/Falcon-TST_Large", trust_remote_code=True
#   )
#       "ant-intl/Falcon-TST_Large", trust_remote_code=True
#   )
#   model.eval()
#   past = torch.from_numpy(y.to_numpy().reshape(1, -1, 1).astype(np.float32)).to(
#       device=model.device, dtype=model.dtype
#   )
#   with torch.no_grad():
#       raw = model.predict(past, forecast_horizon=3, revin=True)
#   expected = raw.detach().float().cpu().numpy().flatten().tolist()
#
_EXPECTED_PREDICTIONS = [441.4195556640625, 441.5230407714844, 457.26953125]


@pytest.mark.skipif(
    not run_test_for_class(FalconTSTForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_falcon_tst_predictions_match_reference():
    """Falcon-TST zero-shot predictions match raw upstream reference outputs."""
    y = load_airline()
    fh = [1, 2, 3]

    forecaster = FalconTSTForecaster()
    y_pred = forecaster.fit(y).predict(fh=fh)

    np.testing.assert_allclose(
        y_pred.to_numpy().flatten(),
        np.asarray(_EXPECTED_PREDICTIONS, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
