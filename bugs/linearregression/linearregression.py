import pandas as pd
import sklearn

from sktime.forecasting.darts import DartsLinearRegressionModel

X, y = sklearn.datasets.make_regression(
    n_samples=1000, n_features=2, noise=0.1, random_state=42
)
# bugs/linearregression/linearregression.py
date_idx = pd.date_range(start="2020-01-01", periods=1000, freq="15min")
X = pd.DataFrame(X, columns=["feature1", "feature2"], index=date_idx)
y = pd.Series(y, index=date_idx)
model = DartsLinearRegressionModel(
    past_covariates=["feature1", "feature2"],
    lags_past_covariates={"feature1": [-1, -2, -3], "feature2": [-1, -2, -3]},
    # assume we have forecasts for feature1
    lags_future_covariates={"feature1": [0, 1]},
    output_chunk_length=1,
)

model.fit(y, X)
