# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""AutoRegressiveWrapper for Forecasters with fixed prediction length."""

__author__ = ["geetu040"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.dependencies import _check_soft_dependencies


class AutoRegressiveWrapper(_BaseGlobalForecaster):
    """
    AutoRegressiveWrapper for Forecasters with Fixed Prediction Length.

    This class acts as a wrapper for existing forecasting models, enabling them
    to perform autoregressive predictions when they are originally designed
    for fixed-length outputs. It addresses the limitations of certain
    forecasters that can only predict a fixed number of future time steps,
    such as the KanForecaster and many reduction-based forecasters.

    By utilizing an autoregressive approach, this wrapper allows the forecaster
    to make predictions iteratively. Each prediction is fed back into the
    model as part of the input for subsequent predictions, which can significantly
    enhance flexibility and efficiency, especially when training time is a
    consideration and when dealing with varying forecasting horizons.

    The `AutoRegressiveWrapper` is particularly useful in scenarios where:
    - The underlying forecaster takes considerable time to train with
      increased output sizes.
    - The forecaster is integrated with probabilistic wrappers (e.g.,
      Conformal Intervals) that require variable forecasting horizons.

    The wrapped forecaster must inherit from `_BaseGlobalForecaster`, ensuring
    that it supports the necessary interface for fitting and predicting.

    Attributes
    ----------
    forecaster : _BaseGlobalForecaster
        An instance of a forecasting model that is compatible with
        the `AutoRegressiveWrapper`. This forecaster must support
        the methods required for fitting and predicting.
    horizon_length : int
        The length of the forecasting horizon, defining how many time
        steps into the future the forecaster should predict with each
        call to the `predict` method.
    aggregate_method : callable
        A method for aggregating predictions over the autoregressive
        iterations. If not specified, defaults to `numpy.mean`.

    Example
    -------
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    >>> from sktime.forecasting.compose import AutoRegressiveWrapper
    >>>
    >>> y = load_airline()
    >>> forecaster = PytorchForecastingTFT(trainer_params={
    ...     "max_epochs": 5,
    ... }) # doctest skip
    >>> wrapper = AutoRegressiveWrapper(
    ...     forecaster=forecaster,
    ...     horizon_length=3, # prediction_length for forecaster
    ...     aggregate_method=np.mean,
    ... ) # doctest skip
    >>>
    >>> wrapper.fit(y) # doctest skip
    >>>
    >>> # forecast multiple forecasting horizons with same trained model
    >>> preds = wrapper.predict(fh=[1, 5]) # doctest skip
    >>> preds = wrapper.predict(fh=[8, 9, 10, 11, 12]) # doctest skip


    """

    _tags = {
        "scitype:y": "both",
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, forecaster, horizon_length, aggregate_method=None):
        self.forecaster = forecaster
        self.horizon_length = horizon_length
        self.aggregate_method = (
            aggregate_method if aggregate_method is not None else np.mean
        )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        # ignore the fh provided fit
        _fh = ForecastingHorizon(range(1, self.horizon_length + 1))

        self.forecaster.fit(y=y, X=X, fh=_fh)

    def _predict(self, fh, X=None, y=None):
        # use fh to find the maximum length to forecast regressively
        max_fh = max(
            *(fh.to_relative(self._cutoff)._values + 1),
            self.horizon_length,
        )

        _y = self._y if y is None else y
        _y = deepcopy(_y)  # make copy because we are going to add values to it

        if y is None and X is not None:
            # special case of non global forecasting
            # here X contains only (future) values
            # however for the interface, X should contain (hist + future) values
            # since this is non-global forecasting, we can use hist from self._X
            X = pd.concat([self._X, X])

        for i in range(max_fh):
            # get sample from X containing historical and future exogenous data
            _x = self._get_x(_y, X, i)

            # forecast for self.horizon_length
            preds = self.forecaster.predict(y=_y, X=_x)

            # use predicted values as history for next iteration
            _y = self._concate_preds(_y, preds)

        # collect forecasting points from _y
        # code works for all indexes
        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = _y.index.get_level_values(-1).map(lambda x: x in absolute_horizons)
        preds = _y.loc[dateindex]

        return preds

    def _get_x(self, _y, X, i):
        # keep rows till this index in X (hist from _y + future from _fh)

        if X is None:
            return None

        if not isinstance(_y.index, pd.MultiIndex):
            end_index = len(_y.index) + self.horizon_length + i
            return X.head(end_index)

        # truncating for Multi-Index
        end_index = len(_y.index.levels[-1]) + self.horizon_length + i

        # levels other than the timestamps (innermost/base level)
        groupby_levels = list(range(len(X.index.names) - 1))

        # truncate X
        _x = X.groupby(level=groupby_levels).apply(lambda x: x.head(end_index))
        _x.index = _x.index.droplevel(groupby_levels)

        return _x

    def _concate_preds(self, _y, preds):
        _y = pd.concat([_y, preds])
        _y = _y.groupby(level=list(range(_y.index.nlevels)))
        _y = _y.aggregate(self.aggregate_method)
        return _y

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_params = []
        if _check_soft_dependencies("pytorch-forecasting", severity="none"):
            import numpy as np

            from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT

            test_params = [
                {
                    "forecaster": PytorchForecastingTFT(
                        **PytorchForecastingTFT.get_test_params()[0],
                    ),
                    "horizon_length": 3,
                    "aggregate_method": np.mean,
                }
            ]
        # TODO: use NaiveForecaster instead of PytorchForecastingTFT
        # as https://github.com/sktime/sktime/pull/6868/files is merged
        return test_params
