# -*- coding: utf-8 -*-
"""Helper Functions for example."""
import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import _PanelToTabularTransformer


def find_maxlag(model_kwargs):
    """Find maximum lag based on provided dictionary."""
    lag_max_list = model_kwargs["lags"]
    window_functions = model_kwargs["window_functions"]
    if len(window_functions) > 0:
        _window_functions = list()
        for func_name, rw_config in window_functions.items():
            func_call, window_shifts, window_sizes = rw_config
            for window_shift in window_shifts:
                for window_size in window_sizes:
                    _window_functions.append(
                        (func_name, func_call, window_shift, window_size)
                    )
        _window_functions = _window_functions
    else:
        _window_functions = list()
    if len(_window_functions) > 0:
        #  lag_kwargs = [{"lag":lag} for lag in lags]
        rw_kwargs = [
            {
                "func_name": window_func[0],
                "func_call": window_func[1],
                "window_shift": window_func[2],
                "window_size": window_func[3],
            }
            for window_func in _window_functions
        ]
    window_max_list = [(i["window_shift"] - 1 + i["window_size"]) for i in rw_kwargs]
    maxlag = max(max(lag_max_list), max(window_max_list))
    # y[y.index <(collect_max-1)] = None
    # y=y[(collect_max-1):]
    return maxlag


class _MVTreeExtractor(_PanelToTabularTransformer):
    """Base adapter class for transformations."""

    def __init__(
        self,
        lags,
        window_functions,
        n_jobs=-1,
        # _is_fitted = False,
        ts_values=None,
    ):

        self.lags = lags
        self.window_functions = window_functions
        self.n_jobs = n_jobs
        # self._is_fitted = _is_fitted
        self.ts_values = ts_values

        super(_MVTreeExtractor, self).__init__()

    # Get extraction parameters
    def fit(self):
        """Fit.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        self : an instance of self
        """
        # check_X(X, coerce_to_pandas=True)

        if len(self.window_functions) > 0:
            _window_functions = list()
            for func_name, rw_config in self.window_functions.items():
                func_call, window_shifts, window_sizes = rw_config
                for window_shift in window_shifts:
                    for window_size in window_sizes:
                        _window_functions.append(
                            (func_name, func_call, window_shift, window_size)
                        )
            self._window_functions = _window_functions
        else:
            self._window_functions = list()

        self._is_fitted = True

        return self


class MVTreeFeatureExtractor(_MVTreeExtractor):
    """Transformer for extracting time series features."""

    def transform(self, Z):
        """Transform X.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series, optional (default=None)

        Returns
        -------
        Xt : pandas DataFrame
        Transformed pandas DataFrame

        """
        # input checks

        self.check_is_fitted()

        Z = Z.copy()

        if (len(self.lags) > 0) or (len(self.window_functions) > 0):
            lag_kwargs = [{"lag": lag} for lag in self.lags]
            rw_kwargs = [
                {
                    "func_name": window_func[0],
                    "func_call": window_func[1],
                    "window_shift": window_func[2],
                    "window_size": window_func[3],
                }
                for window_func in self._window_functions
            ]
            input_kwargs = lag_kwargs + rw_kwargs  # lag_kwargs + rw_kwargs
            grouped = Z.groupby(level=0)["y"]
            df = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_lagged_train_feature)(grouped, **kwargs)
                for kwargs in input_kwargs
            )
            col_names = [o.name for o in df]
            df = pd.concat(df, axis=1)
            df.columns = col_names

        all_features = pd.concat([Z, df], axis=1)

        return all_features


# %%


def compute_lagged_train_feature(
    grouped,
    lag=None,
    func_name=None,
    func_call=None,
    window_shift=None,
    window_size=None,
):
    """Compute lags.

    grouped: pandas.core.groupby.generic.SeriesGroupBy
        Groupby object containing the response variable "y"
        grouped by ts_uid_columns.
    lag: int
        Integer lag value.
    func_name: string
        Name of the rolling window function.
    func_call: function or None
        Callable if a custom function, None otherwise.
    window_shift: int
        Integer window shift value.
    window_size: int
        Integer window size value.
    """
    is_lag_feature = lag is not None
    is_rw_feature = (
        (func_name is not None)
        and (window_shift is not None)
        and (window_size is not None)
    )

    if is_lag_feature and not is_rw_feature:
        feature_values = grouped.shift(lag)
        feature_values.name = f"lag{lag}"
    elif is_rw_feature and not is_lag_feature:
        if func_call is None:
            # native pandas method
            feature_values = grouped.apply(
                lambda x: getattr(
                    x.shift(window_shift).rolling(window_size), func_name
                )()
            )
        else:
            # custom function
            feature_values = grouped.apply(
                lambda x: x.shift(window_shift)
                .rolling(window_size)
                .apply(func_call, raw=True)
            )
        feature_values.name = f"{func_name}{window_size}_shift{window_shift}"
    else:
        raise ValueError("Invalid input parameters.")

    return feature_values
