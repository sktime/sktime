from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.datasets import load_longley
from sktime.split import SingleWindowSplitter, SlidingWindowSplitter
from sktime.forecasting.tests._config import (
    # TEST_N_ITERS,
    TEST_OOS_FHS,
    # TEST_RANDOM_SEEDS,
    # TEST_WINDOW_LENGTHS_INT,
)
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.model_selection import (
    TuneForecastingOptunaCV,
    ForecastingGridSearchCV,
)
import pandas as pd
import optuna
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", None)

CVs = [
    *[SingleWindowSplitter(fh=fh) for fh in TEST_OOS_FHS],
    SlidingWindowSplitter(fh=1, initial_window=12, step_length=3),
]
params_distributions = {
    # "forecaster": [NaiveForecaster()],
    "forecaster__window_length": optuna.distributions.IntDistribution(2, 6),
    "forecaster__strategy": optuna.distributions.CategoricalDistribution(
        ("drift", "last", "mean")
    ),
    "imputer__method": optuna.distributions.CategoricalDistribution(("mean", "median")),
}
params_grid = {
    # "forecaster": [NaiveForecaster()],
    "forecaster__window_length": [2, 3, 4, 5, 6],
    "forecaster__strategy": ["drift", "last", "mean"],
    "imputer__method": ["mean", "median"],
}
cv = CVs[-1]
# y = load_airline()
y, X = load_longley()
pipe = TransformedTargetForecaster(
    steps=[("imputer", Imputer()), ("forecaster", NaiveForecaster())]
)
sscv = TuneForecastingOptunaCV(
    forecaster=pipe,
    param_grid=params_distributions,
    cv=cv,
    n_evals=10,
)
# sscv = ForecastingGridSearchCV(
#     forecaster=pipe,
#     param_grid=params_grid,
#     cv=cv,
# )

sscv.fit(y, X)
print(pd.DataFrame(sscv.cv_results_))
print(pd.DataFrame(sscv.cv_results_)[['params','rank_test_MeanAbsolutePercentageError', 'value']])
print(cv)
