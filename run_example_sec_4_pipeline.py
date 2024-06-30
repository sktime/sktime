import warnings

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingOptunaSearchCV,
    ForecastingSkoptSearchCV,
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
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import Deseasonalizer, Detrender

forecaster = TransformedTargetForecaster(
    steps=[
        ("detrender", Detrender()),
        ("deseasonalizer", Deseasonalizer()),
        ("minmax", TabularToSeriesAdaptor(MinMaxScaler((1, 10)))),
        ("power", TabularToSeriesAdaptor(PowerTransformer())),
        ("scaler", TabularToSeriesAdaptor(RobustScaler())),
        ("forecaster", ExponentialSmoothing()),
    ]
)

# using dunder notation to access inner objects/params as in sklearn
param_grid = {
    # deseasonalizer
    "deseasonalizer__model": optuna.distributions.CategoricalDistribution(
        ("multiplicative", "additive")
    ),
    # power
    "power__transformer__method": optuna.distributions.CategoricalDistribution(
        ("yeo-johnson", "box-cox")
    ),
    "power__transformer__standardize": optuna.distributions.CategoricalDistribution(
        (True, False)
    ),
    # forecaster
    "forecaster__sp": optuna.distributions.IntDistribution(4, 12),
    "forecaster__seasonal": optuna.distributions.CategoricalDistribution(
        ("add", "mul")
    ),
    "forecaster__trend": optuna.distributions.CategoricalDistribution(("add", "mul")),
    "forecaster__damped_trend": optuna.distributions.CategoricalDistribution(
        (True, False)
    ),
}
param_grid = {
    # deseasonalizer
    "deseasonalizer__model": ["multiplicative", "additive"],
    # power
    "power__transformer__method": ["yeo-johnson", "box-cox"],
    "power__transformer__standardize": [True, False],
    # forecaster
    "forecaster__sp": [4, 6, 12],
    "forecaster__seasonal": ["add", "mul"],
    "forecaster__trend": ["add", "mul"],
    "forecaster__damped_trend": [True, False],
}

gscv = ForecastingGridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid,
    cv=cv,
    verbose=1,
    scoring=MeanSquaredError(square_root=True),  # set custom scoring function
    n_evals=100,
)
gscv.fit(y_train)
y_pred = gscv.predict(fh=fh)
print(f"{gscv.best_params_=}")
# {'deseasonalizer__model': 'additive', 'forecaster__damped_trend': False, 'forecaster__seasonal': 'add', 'forecaster__sp': 6, 'forecaster__trend': 'add', 'power__transformer__method': 'yeo-johnson', 'power__transformer__standardize': True}
# {'deseasonalizer__model': 'additive', 'power__transformer__method': 'yeo-johnson', 'power__transformer__standardize': True, 'forecaster__sp': 6, 'forecaster__seasonal': 'add', 'forecaster__trend': 'add', 'forecaster__damped_trend': False}
