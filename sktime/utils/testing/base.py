import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.reduction import ReducedTabularRegressorMixin
from sktime.forecasting.reduction import ReducedTimeSeriesRegressorMixin
from sktime.transformers.compose import Tabulariser

# look up table for estimators which require arguments during constructions,
# links base classes with the default constructor arguments
REGRESSOR = LinearRegression()

DEFAULT_INSTANTIATIONS = {
    ReducedTabularRegressorMixin: {"regressor": REGRESSOR},
    ReducedTimeSeriesRegressorMixin: {"regressor": make_pipeline(Tabulariser(), REGRESSOR)}
}


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters) > 0:
        # if estimator requires parameters for construction,
        # set default ones for testing
        if issubclass(Estimator, BaseForecaster):

            kwargs = {}
            for base in Estimator.__bases__:
                if base in DEFAULT_INSTANTIATIONS:
                    kwargs = DEFAULT_INSTANTIATIONS[base]

            if not kwargs:
                raise ValueError(f"no default instantiation has been found "
                                 f"for estimator: {Estimator}")

        else:
            raise NotImplementedError()

        estimator = Estimator(**kwargs)

    else:
        # construct without kwargs if no parameters are required
        estimator = Estimator()

    return estimator


def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame([[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


