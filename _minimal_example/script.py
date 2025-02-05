#%% md
# ## Example Case with sktime data
# %%
from sklearn.pipeline import make_pipeline
from sktime.forecasting.compose._reduce import (
    RecursiveReductionForecaster,
)
from xgboost import XGBRegressor
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingOptunaSearchCV,
)
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.utils.estimator_checks import check_estimator

import optuna

# parallel_config = {
#     "backend:parallel": "joblib",
#     "backend:parallel:params": {"backend": "loky", "n_jobs": 1},  # deactivate parallel here
# }

lags = 12

params_xgb = {
    "random_state": 42,
}

regressor = make_pipeline(
    XGBRegressor(**params_xgb),
)

model_xgb = RecursiveReductionForecaster(
    estimator=regressor,
    impute_method="bfill",
    pooling="local",
    window_length=lags,
)

pipe_forecast = model_xgb
#%%
from utils import load_stallion

_, y = load_stallion(as_period=True)
#%%
# Filter the data to reduce the dataset size
y = y.reset_index()
y = y[y['agency'].str.match(r'Agency_0[1-2]$')]
y = y[y['sku'].str.match(r'SKU_0[1-3]$')]
y = y[y['date'] > '2013-12']
y = y.set_index(["agency", "sku", "date"])
#%%
from sktime.transformations.hierarchical.aggregate import Aggregator

agg = Aggregator(flatten_single_levels=False)
data_agg = agg.reset().fit_transform(y)
#%%
from sktime. split import temporal_train_test_split

y_train, y_test = temporal_train_test_split(data_agg, test_size=18)
test_fh = y_test.index.get_level_values(-1).unique()
#%%
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
import numpy as np

# parameter_search_space_optuna = {
#     "estimator__xgbregressor__n_estimators": optuna.distributions.IntDistribution(10, 3000),
# }

parameter_search_space_grid = {
    "estimator__xgbregressor__n_estimators": list(range(10, 3000, 10)),  # grid
}

# parameter_search_space_rrf = {
#     "forecaster__forecaster__estimator__xgbregressor__n_estimators": optuna.distributions.IntDistribution(10, 3000),
# }

fh = 18
fold = 4
step_length=3
y_size = len(y_train.index.get_level_values(-1).unique())
single_fold_length = y_size - (fold - 1) * step_length - fh
fold_strategy =  ExpandingWindowSplitter(
                fh=np.arange(1, fh + 1), initial_window=single_fold_length, step_length=step_length
            )

# sampler = optuna.samplers.TPESampler(seed=42)

# grid = ForecastingGridSearchCV(
#     forecaster=pipe_forecast,
#     param_grid=parameter_search_space_grid,
#     cv=fold_strategy,
#     error_score="raise",
#     scoring=MeanAbsoluteError(),
# )


# htcv = ForecastingOptunaSearchCV(
#     forecaster=pipe_forecast,
#     param_grid=parameter_search_space_optuna,
#     cv=fold_strategy,
#     n_evals=5,
#     scoring=MeanAbsoluteError(),
#     error_score="raise",
#     sampler=sampler,
# )
#%% md
# ## Show Version
#%%
import pandas as pd
#pd.show_versions()
#%% md
# ## Check Estimator
#%%
# vscode crashes
#check_est_before_fit_grid = check_estimator(grid)
##%%
#check_est_before_fit_optuna = check_estimator(htcv)
##%%
#check_est_before_fit = check_estimator(model_xgb)
#%% md
# ## Fit
#%%
# fit works for XGBoost and RRF
only_rff_xgb_fitted = model_xgb.fit(
   y=y_train,
   fh=test_fh,
)
#%%
# fit fails
# TypeError: Cannot convert input [('Agency_01', 'SKU_01', Period('2013-01', 'M'))] of type <class 'tuple'> to Timestamp
#optuna_fitted = htcv.fit(
#    y=y_train,
#    fh=test_fh,
#)
#%%
# fit fails
# TypeError: Cannot convert input [('Agency_01', 'SKU_01', Period('2013-01', 'M'))] of type <class 'tuple'> to Timestamp

# grid_fitted = grid.fit(
#     y=y_train,
#     fh=test_fh,
# )
#%% md
# ## predict
#%%
# fails
# TypeError: Cannot convert input [('Agency_01', 'SKU_01', Period('2013-01', 'M'))] of type <class 'tuple'> to Timestamp
print(f"Preds only RRF: {only_rff_xgb_fitted.predict(fh=18)}")
#%%
# optuna_fitted.predict(fh=test_fh)
#%%
# grid_fitted.predict(fh=test_fh)