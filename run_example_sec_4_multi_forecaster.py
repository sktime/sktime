import warnings

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingOptunaSearchCV,
)
from sktime.performance_metrics.forecasting import MeanSquaredError

warnings.simplefilter(action="ignore", category=FutureWarning)
import optuna

from sktime.forecasting.base import ForecastingHorizon
from sktime.split import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.utils.plotting import plot_series, plot_windows

y = load_shampoo_sales()
y_train, y_test = temporal_train_test_split(y=y, test_size=6)


fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
    cutoff=y_train.index[-1]
)


cv = ExpandingWindowSplitter(fh=fh, initial_window=24, step_length=1)

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler

from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import STLForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import Deseasonalizer, Detrender

forecaster = TransformedTargetForecaster(
    steps=[
        ("detrender", Detrender()),
        ("deseasonalizer", Deseasonalizer()),
        ("scaler", TabularToSeriesAdaptor(RobustScaler())),
        ("minmax2", TabularToSeriesAdaptor(MinMaxScaler((1, 10)))),
        ("forecaster", NaiveForecaster()),
    ]
)

param_grid = {
    "scaler__transformer__with_scaling": optuna.distributions.CategoricalDistribution(
        (True, False)
    ),
    "forecaster": optuna.distributions.CategoricalDistribution(
        (STLForecaster(), ThetaForecaster())
    ),
}
# [
# {
#     "scaler__transformer__with_scaling": optuna.distributions.CategoricalDistribution(
#         (True, False)
#     ),
#     # "forecaster": optuna.distributions.CategoricalDistribution(
#     #     (NaiveForecaster(), NaiveForecaster())
#     # ),
# },
# ,
# ]
# param_grid = [
#     {
#         "scaler__transformer__with_scaling": [True, False],
#         "forecaster": [NaiveForecaster()],
#         "forecaster__strategy": ["drift", "last", "mean"],
#         "forecaster__sp": [4, 6, 12],
#     },
#     {
#         "scaler__transformer__with_scaling": [True, False],
#         "forecaster": [STLForecaster(), ThetaForecaster()],
#         "forecaster__sp": [4, 6, 12],
#     },
# ]
gscv = ForecastingOptunaSearchCV(
    forecaster=forecaster,
    param_grid=param_grid,
    cv=cv,
    n_evals=10,
)
gscv.fit(y)
print(f"{gscv.best_params_=}")
# gscv.best_params_={'forecaster': NaiveForecaster(sp=4), 'forecaster__sp': 4, 'forecaster__strategy': 'last', 'scaler__transformer__with_scaling': True}
# {'deseasonalizer__model': 'additive', 'power__transformer__method': 'yeo-johnson', 'power__transformer__standardize': True, 'forecaster__sp': 6, 'forecaster__seasonal': 'add', 'forecaster__trend': 'add', 'forecaster__damped_trend': False}
