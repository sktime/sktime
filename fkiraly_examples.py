# -*- coding: utf-8 -*-
"""Example application of global forecasting."""
# %%
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from global_fc_helpers import MVTimeSeriesSplit, mvts_cv

# from sktime.datatypes._panel._convert import _get_time_index
from sktime.forecasting.compose import ForecastingPipeline, make_reduction

# from sktime.forecasting.model_selection import (
#     ExpandingWindowSplitter,
#     ForecastingGridSearchCV,
# )
from sktime.transformations.series.summarize import WindowSummarizer

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

# %%
# Load M5 Data and prepare
data = pd.read_pickle("daten.jay")
data.rename(
    columns={"value": "y", "ds": "timepoints", "ts_id": "instances"}, inplace=True
)

data["timepoints"] = pd.to_datetime(data["timepoints"], format="%Y-%m-%d")
data.sort_values(["instances", "timepoints"], inplace=True)

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
data = data[data["instances"].isin(data["instances"].unique()[:2])]
data.sort_values(["instances", "timepoints"], inplace=True)
data = data.groupby("instances").tail(40)
# %%

# Build Multiindex format
dateline = pd.to_datetime(data["timepoints"].unique())
dateline.freq = "D"

data.drop("combine", inplace=True, axis=1)

tsids = data["instances"].unique()

mi = pd.MultiIndex.from_product([tsids, dateline], names=["instances", "timepoints"])


y = pd.DataFrame(data["y"].values, index=mi, columns=["y"])
X = pd.DataFrame(
    data.drop(["y", "instances", "timepoints"], axis=1).values,
    index=mi,
    columns=data.drop(["y", "instances", "timepoints"], axis=1).columns,
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
    "lag_config": {
        "lag": ["lag", [[1, 0]]],
        "mean": ["mean", [[3, 0], [12, 0]]],
        "std": ["std", [[4, 0]]],
    }
}


# %%
# from sktime.datasets import load_airline
# from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.model_selection import temporal_train_test_split
# from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
# from sktime.utils.plotting import plot_series

# # data loading for illustration (see section 1 for explanation)
# y = load_airline()
# y_train, y_test = temporal_train_test_split(y, test_size=36)
# fh = ForecastingHorizon(y_test.index, is_relative=False)

# from sklearn.neighbors import KNeighborsRegressor

# from sktime.forecasting.compose import make_reduction

# regressor = KNeighborsRegressor(n_neighbors=1)
# forecaster = make_reduction(regressor, window_length=15, strategy="recursive")

# forecaster.fit(y_train)
# y_pred = forecaster.predict(fh)
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
# mean_absolute_percentage_error(y_pred, y_test)

# %%

# from sktime.forecasting.base import ForecastingHorizon

# Example 1

kwargs = {
    "lag_config": {
        "lag": ["lag", [[1, 0], [2, 0], [3, 0]]],
    }
}

regressor = make_pipeline(
    RandomForestRegressor(),
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs)],
    window_length=None,
)

forecaster.fit(X=X_train, y=y_train, fh=2)

y_pred = forecaster.predict(X=X_test, fh=2)
# print(y_pred)


# Example 3 - None for transformers does not work yet, but easy to adjust
#  - no window needed since number of lag of y in this example

regressor = make_pipeline(
    RandomForestRegressor(),
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=None,
    window_length=None,
)

pipe = ForecastingPipeline(
    steps=[
        (
            "a",
            WindowSummarizer(n_jobs=1, target_cols=["event_type_1", "snap"], **kwargs),
        ),
        ("forecaster", forecaster),
    ]
)


# Example 4 - I also included lags of y to make example more interesting

kwargs_y = {
    "lag_config": {
        "lag": ["lag", [[1, 0], [2, 0], [3, 0]]],
    }
}

kwargs_x = {
    "lag_config": {
        "mean": ["mean", [[12, 12], [24, 12]]],
    }
}


regressor = make_pipeline(
    RandomForestRegressor(),
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs_y)],
    window_length=None,
)

pipe = ForecastingPipeline(
    steps=[
        (
            "a",
            WindowSummarizer(
                n_jobs=1, target_cols=["event_type_1", "snap"], **kwargs_x
            ),
        ),
        ("forecaster", forecaster),
    ]
)


pipe_return = pipe.fit(y_train, X_train)
y_pred1 = pipe_return.predict(fh=1, X=X_test)


forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs)],
    window_length=None,
)

forecaster.fit(X=X_train, y=y_train, fh=2)

y_pred = forecaster.predict(X=X_test, fh=2)
# print(y_pred)
# # %%
# # Cross Validation
# kwargs_cv = {
#     "functions": {
#         "lag": ["lag", [[1, 1], [3, 1], [5, 1]]],
#         "mean": ["mean", [[1, 1], [5, 1], [3, 2]]],
#         "median": ["median", [[1, 2], [2, 3]]],
#         "std": ["std", [[1, 2], [5, 2], [4, 2]]],
#     }
# }

# # Window length should correspond dynamically to transformer choices
# # How to implement that?

# # Ultimate goal: have a complexity argument
# taking account the frequency of time series
# # Automatically generate a dictionary

# forecaster_param_grid = {
#     "window_length": [None],
#     "transformers": [
#         LaggedWindowSummarizer(**kwargs),
#         LaggedWindowSummarizer(**kwargs_cv),
#     ],
# }

# regressor_param_grid = {"random_state": [8, 12]}

# # include get:wind
# y_length = len(_get_time_index(y_train))

# cv = ExpandingWindowSplitter(initial_window=15)

# regressor = GridSearchCV(
#     RandomForestRegressor(),
#     param_grid=regressor_param_grid,
#     scoring="neg_mean_absolute_error",
# )

# forecaster = make_reduction(
#     regressor,
#     scitype="tabular-regressor",
#     transformers=LaggedWindowSummarizer(**kwargs),
#     window_length=None,
# )

# gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=forecaster_param_grid)

# # %%

# gscv.fit(X=X_train, y=y_train, fh=2)
# y_pred_cv = gscv.predict(X=X_test, fh=2)

# # mean absolute precentage error: How is it calcualted.
# # Should we allow for a composite calculation
# # mean_absolute_percentage_error(y_pred, y_test)

# gscv.best_params_
# print(gscv.best_params_)
# print(gscv.best_forecaster_.estimator_.best_params_)
# print(y_pred_cv)
