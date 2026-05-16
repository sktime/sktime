"""Fix attempt: correct covariate split for sktime DartsLinearRegressionModel.

Note: the correct params below WORK with darts natively (see _darts_native.py)
but still fail through the sktime adapter due to a bug in convert_exogenous_dataset
which doesn't properly route features based on lags_future_covariates keys.
The adapter bug is in sktime/forecasting/base/adapters/_darts.py line 220-226.
"""
import sklearn
import pandas as pd
from sktime.forecasting.darts import DartsLinearRegressionModel


X, y = sklearn.datasets.make_regression(
    n_samples=1000, n_features=2, noise=0.1, random_state=42
)
date_idx = pd.date_range(start="2020-01-01", periods=1000, freq="15min")
X = pd.DataFrame(X, columns=["feature1", "feature2"], index=date_idx)
y = pd.Series(y, index=date_idx)

# feature1 is known into the future -> future covariate with both past and future lags
# feature2 is only known historically -> past covariate
model = DartsLinearRegressionModel(
    past_covariates=["feature2"],
    lags_past_covariates={"feature2": [-1, -2, -3]},
    lags_future_covariates={"feature1": [-1, -2, -3, 0, 1]},
    output_chunk_length=1,
)

model.fit(y, X)
print("SUCCESS: fit with both past and future covariates works")
