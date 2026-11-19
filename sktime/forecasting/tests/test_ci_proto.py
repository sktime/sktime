"""Golden-output regression tests for CIProtoForecaster.

Reference values come from continuousintelligence 0.1.1, using the pinned
ci-proto checkpoint on CPU. The standalone upstream scripts are included below.
Estimator interface conformance is covered by check_estimator.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ci_proto import CIProtoForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(CIProtoForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

REVISION = "ee3031351a151e3f6febf8c62b53e6ea1008bef6"
# Regenerate with continuousintelligence==0.1.1, without using this adapter:
#
# import continuousintelligence as ci
# import numpy as np
# import pandas as pd
# import torch
# from sktime.datasets import load_airline
#
# model = ci.load_model(
#     "ContinuousIntelligence/ci-proto", device="cpu",
#     revision="ee3031351a151e3f6febf8c62b53e6ea1008bef6",
# )
# y = load_airline().iloc[:-12]
# dates = pd.period_range(y.index[0], periods=len(y) + 3, freq="M").to_timestamp()
# output = model.forecast(
#     torch.as_tensor(y.to_numpy()[None, :], dtype=torch.float32), horizon=3,
#     timestamps=torch.as_tensor(dates.as_unit("ns").asi8.copy()),
#     freq=dates.freqstr,
# )
# print(output[0, 4].tolist())
EXPECTED_AIRLINE = [406.0, 392.0, 463.0]
# After the imports and model loading above, generate the covariate reference:
#
# rng = np.random.default_rng(0)
# history = rng.normal(size=(2, 32))
# history[0, 5] = np.nan
# past = rng.normal(size=(1, 32))
# known = rng.normal(size=(1, 36))
# output = model.forecast(
#     torch.as_tensor(history, dtype=torch.float32), horizon=4,
#     past_covariates=torch.as_tensor(past, dtype=torch.float32),
#     future_covariates=torch.as_tensor(known, dtype=torch.float32),
# )
# print(output[:, [0, 4, 8]][:, :, [0, 3]].permute(2, 0, 1).reshape(2, 6).tolist())
EXPECTED_COVARIATES = [
    [-1.25473404, -0.21862656, 0.86096776, -1.19476199, 0.14691001, 1.41560102],
    [-1.33659160, -0.21385482, 0.90504491, -1.17592812, 0.21965551, 1.52107024],
]
# To reproduce an update without new X, continue the upstream script above:
#
# past[:, 28:] = np.nan
# known[:, 28:32] = np.nan
# output = model.forecast(
#     torch.as_tensor(history, dtype=torch.float32), horizon=4,
#     past_covariates=torch.as_tensor(past, dtype=torch.float32),
#     future_covariates=torch.as_tensor(known, dtype=torch.float32),
# )
# print(output[:, [0, 4, 8]][:, :, [0, 3]].permute(2, 0, 1).reshape(2, 6).tolist())
EXPECTED_UPDATED_WITHOUT_X = [
    [-1.26103079, -0.22571039, 0.84837437, -1.23619628, 0.11606968, 1.39676738],
    [-1.34918499, -0.22079106, 0.90504491, -1.20982897, 0.20541248, 1.53237057],
]


@pytest.mark.parametrize("datetime_index", [False, True])
def test_ci_proto_airline_matches_source_reference(datetime_index):
    """The median forecast matches the upstream airline forecast."""
    y = load_airline().iloc[:-12]
    if datetime_index:
        y.index = y.index.to_timestamp()
    forecaster = CIProtoForecaster(device="cpu", revision=REVISION)
    if datetime_index:
        forecaster.fit(y.iloc[:-1])
        latest = y.iloc[-1:].copy()
        latest.index = pd.DatetimeIndex([y.index[-1]])
        forecaster.update(latest, update_params=False)
    else:
        forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True, freq=y.index.freq)
    prediction = forecaster.predict(fh=fh)

    np.testing.assert_allclose(
        prediction.to_numpy().ravel(), EXPECTED_AIRLINE, rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize(
    "update_params, remember_data, extra_future_steps, update_covariates",
    [
        (None, False, 0, True),
        (False, False, 0, True),
        (True, False, 2, True),
        (False, True, 2, True),
        (True, True, 0, True),
        (False, False, 0, False),
    ],
)
def test_ci_proto_quantiles_with_covariates_match_source_reference(
    update_params, remember_data, extra_future_steps, update_covariates
):
    """Sparse marginal quantiles match upstream with mixed covariates."""
    rng = np.random.default_rng(0)
    history = rng.normal(size=(2, 32))
    history[0, 5] = np.nan
    past = rng.normal(size=(1, 32))
    known = rng.normal(size=(1, 36))
    y = pd.DataFrame(history.T, columns=["a", "b"])
    X = pd.DataFrame({"past": past[0], "known": known[0, :32]})
    X_future = pd.DataFrame({"known": known[0, 32:]}, index=pd.RangeIndex(32, 36))
    if extra_future_steps:
        X_future = pd.concat(
            [X_future, pd.DataFrame({"known": [0.0, 0.0]}, index=[36, 37])]
        )
    forecaster = CIProtoForecaster(device="cpu", revision=REVISION)
    forecaster.set_config(remember_data=remember_data)
    if update_params is None:
        forecaster.fit(y, X=X)
    else:
        forecaster.fit(y.iloc[:28], X=X.iloc[:28])
        forecaster.update(
            y.iloc[28:],
            X=X.iloc[28:] if update_covariates else None,
            update_params=update_params,
        )
    distribution = forecaster.predict_proba(fh=[1, 4], X=X_future)
    quantiles = distribution.quantile([0.1, 0.5, 0.9])

    np.testing.assert_allclose(
        quantiles.to_numpy(),
        EXPECTED_COVARIATES if update_covariates else EXPECTED_UPDATED_WITHOUT_X,
        rtol=1e-5,
        atol=1e-4,
    )
