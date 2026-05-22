"""Workaround: duplicate future covariate columns with _future suffix.

CAVEAT: convert_exogenous_dataset returns (known, unknown) but _fit unpacks
as (unknown, known) — so past/future covariates are SWAPPED (existing bug).
To work around BOTH bugs, we put the future covariates in past_covariates
so the swap sends them to the correct darts argument.
"""

import pandas as pd
import sklearn

from sktime.forecasting.darts import DartsLinearRegressionModel

X, y = sklearn.datasets.make_regression(
    n_samples=1000, n_features=2, noise=0.1, random_state=42
)
date_idx = pd.date_range(start="2020-01-01", periods=1000, freq="15min")
X = pd.DataFrame(X, columns=["feature1", "feature2"], index=date_idx)
y = pd.Series(y, index=date_idx)

# Duplicate feature1 as feature1_future so it can be routed to future covariates
X["feature1_future"] = X["feature1"]

# Because of the swap bug, past_covariates actually becomes future_covariates
# in darts, and the remaining columns become past_covariates.
# So we put the FUTURE covariate column in past_covariates to counteract the swap.
model = DartsLinearRegressionModel(
    past_covariates=["feature1_future"],
    lags_past_covariates={"feature1": [-1, -2, -3], "feature2": [-1, -2, -3]},
    lags_future_covariates={"feature1_future": [0, 1]},
    output_chunk_length=1,
)

model.fit(y, X)
print("SUCCESS: workaround with _future suffix works")
