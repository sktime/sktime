"""Golden-output regression tests for CIProtoForecaster.

Reference values come from continuousintelligence 0.1.1, using the pinned
ci-proto checkpoint on CPU. For the airline data, pass the training values
as a single row to the upstream forecast method and take the median.
For the multivariate case, pass history, past and known as float32 tensors
with horizon=4, then select steps 1 and 4 and quantiles 0.1, 0.5 and 0.9.
Estimator interface conformance is covered by check_estimator.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.ci_proto import CIProtoForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(CIProtoForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

REVISION = "ee3031351a151e3f6febf8c62b53e6ea1008bef6"
EXPECTED_AIRLINE = [408.0, 392.0, 455.0]
EXPECTED_COVARIATES = [
    [-1.25473404, -0.21862656, 0.86096776, -1.19476199, 0.14691001, 1.41560102],
    [-1.33659160, -0.21385482, 0.90504491, -1.17592812, 0.21965551, 1.52107024],
]


def test_ci_proto_airline_matches_source_reference():
    """The median forecast matches the upstream airline forecast."""
    y = load_airline().iloc[:-12]
    forecaster = CIProtoForecaster(device="cpu", revision=REVISION).fit(y)
    prediction = forecaster.predict(fh=[1, 2, 3])

    np.testing.assert_allclose(
        prediction.to_numpy().ravel(), EXPECTED_AIRLINE, rtol=1e-5, atol=1e-4
    )


def test_ci_proto_quantiles_with_covariates_match_source_reference():
    """Sparse marginal quantiles match upstream with mixed covariates."""
    rng = np.random.default_rng(0)
    history = rng.normal(size=(2, 32))
    history[0, 5] = np.nan
    past = rng.normal(size=(1, 32))
    known = rng.normal(size=(1, 36))
    y = pd.DataFrame(history.T, columns=["a", "b"])
    X = pd.DataFrame({"past": past[0], "known": known[0, :32]})
    X_future = pd.DataFrame({"known": known[0, 32:]}, index=pd.RangeIndex(32, 36))
    forecaster = CIProtoForecaster(device="cpu", revision=REVISION).fit(y, X=X)
    distribution = forecaster.predict_proba(fh=[1, 4], X=X_future)
    quantiles = distribution.quantile([0.1, 0.5, 0.9])

    np.testing.assert_allclose(
        quantiles.to_numpy(), EXPECTED_COVARIATES, rtol=1e-5, atol=1e-4
    )
