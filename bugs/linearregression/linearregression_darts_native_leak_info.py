import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel

from sktime.forecasting.model_selection import temporal_train_test_split

np.random.seed(42)
date_idx = pd.date_range(start="2020-01-01", periods=1000, freq="15min")
X = pd.DataFrame(
    {
        "feature1": np.random.randint(1, 1000, size=1000),
    },
    index=date_idx,
)

# y[t] = X["feature1"][t-1]  (shift(1): y is 1 step BEHIND feature1)
# → when predicting y[t+1] = X[t], past_cov at lag=-1 accesses X[t]: perfect predictor
# → no future covariate needed; info leaks purely from a PAST covariate
y = pd.Series(X.shift(1)["feature1"].values, index=date_idx)

# drop first row (NaN from shift(1))
y = y[y.notnull()]
X = X.loc[y.index]

y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=0.2)

X_full = pd.concat([X_train, X_test])

target = TimeSeries.from_dataframe(pd.DataFrame({"y": y_train}, index=y_train.index))
past_cov_train = TimeSeries.from_dataframe(X_train)

model = LinearRegressionModel(
    lags=1,
    lags_past_covariates={"feature1": [-1]},
    output_chunk_length=1,
)

model.fit(target, past_covariates=past_cov_train)

past_cov_full = TimeSeries.from_dataframe(X_full)

y_pred_ts = model.predict(
    n=1,
    past_covariates=past_cov_full,
)

y_pred_value = y_pred_ts.values().flatten()[0]
y_true_value = y_test.iloc[0]

mae = abs(y_pred_value - y_true_value)
print(mae)
