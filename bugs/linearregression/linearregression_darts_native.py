"""Same scenario as linearregression.py but using darts directly to show it works."""
import sklearn
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel


X, y = sklearn.datasets.make_regression(
    n_samples=1000, n_features=2, noise=0.1, random_state=42
)
date_idx = pd.date_range(start="2020-01-01", periods=1000, freq="15min")

target = TimeSeries.from_dataframe(pd.DataFrame({"y": y}, index=date_idx))
past_cov = TimeSeries.from_dataframe(
    pd.DataFrame({"feature2": X[:, 1]}, index=date_idx)
)
future_cov = TimeSeries.from_dataframe(
    pd.DataFrame({"feature1": X[:, 0]}, index=date_idx)
)

model = LinearRegressionModel(
    lags=3,
    lags_past_covariates={"feature2": [-1, -2, -3]},
    lags_future_covariates={"feature1": [-1, -2, -3, 0, 1]},
    output_chunk_length=1,
)

model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
print("SUCCESS: darts native fit with both past and future covariates works")
