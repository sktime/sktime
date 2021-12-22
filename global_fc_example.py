# -*- coding: utf-8 -*-
"""Example application of global forecasting."""
# %%
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from global_fc_helpers import MVTimeSeriesSplit, mvts_cv
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.transformations.panel.window_summarizer import (
    LaggedWindowSummarizer,
    find_maxlag,
)

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

# %%
# Load M5 Data and prepare
data = pd.read_pickle("daten.jay")
data.rename(columns={"value": "y", "ds": "Period"}, inplace=True)

data["Period"] = pd.to_datetime(data["Period"], format="%Y-%m-%d")
data.sort_values(["ts_id", "Period"], inplace=True)

encode_cols = [
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "no_stock_days",
]

id_encoder = OrdinalEncoder()
for i in encode_cols:
    id_encoder.fit(data.loc[:, [i]])
    data[i] = id_encoder.transform(data.loc[:, [i]])

# Form subsets for faster training
data = data[data["ts_id"].isin(data["ts_id"].unique()[:1])]
data.sort_values(["ts_id", "Period"], inplace=True)
data = data.groupby("ts_id").tail(40)
# %%

# Build Multiindex format
dateline = pd.to_datetime(data["Period"].unique())
dateline.freq = "D"

tsids = data["ts_id"].unique()

mi = pd.MultiIndex.from_product([tsids, dateline], names=["ts_id", "Period"])


y = pd.DataFrame(data["y"].values, index=mi, columns=["y"])
X = pd.DataFrame(
    data.drop(["y", "ts_id", "Period"], axis=1).values,
    index=mi,
    columns=data.drop(["y", "ts_id", "Period"], axis=1).columns,
)

# %%

tscv = MVTimeSeriesSplit(n_splits=3, test_size=3)

X_train, X_test, y_train, y_test = mvts_cv(X, y, tscv)


# %%
# window argument: list of how many features we want
# Item [1, 1] means: shift by one, window of one
# Leonidas suggestion:
# "lag": {["lag", [[1, 1], [5, 1], [7, 1]]]},

# kwargs = {
#     "functions": {
#         "lag": {"func": "lag", "window": [[1, 1], [5, 1], [7, 1]]},
#         "mean": {"func": "mean", "window": [[1, 2], [2, 2], [4, 2]]},
#         "median": {"func": "median", "window": [[1, 2], [2, 2], [4, 2]]},
#         "std": {"func": "std", "window": [[1, 3], [2, 3]]},
#     }
# }

kwargs = {
    "functions": {
        "lag": ["lag", [[1, 1], [4, 1], [7, 1]]],
        "mean": ["mean", [[1, 1], [5, 1], [3, 2]]],
        "median": ["median", [[1, 2], [2, 3]]],
        "std": ["std", [[1, 2], [5, 2], [4, 2]]],
    }
}

# Should it really be 8 or is 7 enough
window_length = find_maxlag(kwargs)

# %%

regressor = make_pipeline(
    RandomForestRegressor(),
)

# Danbartl: Window Length is redundant
# Leonidas: test and make it work for univariate even without multiindex
# Test how it works with LaggedWindowSummarizer
# > should be possible to use in a ForecastingPipeline for X arguments

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[LaggedWindowSummarizer(**kwargs)],
    window_length=None,
)

forecaster.fit(X=X_train, y=y_train, fh=2)

y_pred = forecaster.predict(X=X_test, fh=2)

# %%
# Cross Validation
kwargs_cv = {
    "functions": {
        "lag": ["lag", [[1, 1], [3, 1], [5, 1]]],
        "mean": ["mean", [[1, 1], [5, 1], [3, 2]]],
        "median": ["median", [[1, 2], [2, 3]]],
        "std": ["std", [[1, 2], [5, 2], [4, 2]]],
    }
}

# Window length should correspond dynamically to transformer choices
# How to implement that?

# Ultimate goal: have a complexity argument taking account the frequency of time series
# Automatically generate a dictionary

forecaster_param_grid = {
    "window_length": [8],
    "transformers": [
        LaggedWindowSummarizer(**kwargs),
        LaggedWindowSummarizer(**kwargs_cv),
    ],
}

regressor_param_grid = {"random_state": [8, 12]}

# include get:wind
y_length = len(
    y_train.xs(y_train.index.get_level_values("ts_id")[0], level="ts_id").index
)

cv = ExpandingWindowSplitter(initial_window=15)

regressor = GridSearchCV(
    RandomForestRegressor(),
    param_grid=regressor_param_grid,
    scoring="neg_mean_absolute_error",
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[LaggedWindowSummarizer(**kwargs)],
    window_length=None,
)

gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=forecaster_param_grid)

# %%

gscv.fit(X=X_train, y=y_train, fh=2)
y_pred_cv = gscv.predict(X=X_test, fh=2)

# mean absolute precentage error: How is it calcualted.
# Should we allow for a composite calculation
# mean_absolute_percentage_error(y_pred, y_test)

gscv.best_params_
# print(gscv.best_params_)
# print(gscv.best_forecaster_.estimator_.best_params_)
# print(y_pred_cv)
