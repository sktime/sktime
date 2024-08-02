# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""AutoRegressiveWrapper for Forecasters with fixed prediction length."""

__author__ = ["geetu040"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.dependencies import _check_soft_dependencies


class AutoRegressiveWrapper(_BaseGlobalForecaster):
    """AutoRegressiveWrapper."""

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

        for i in range(max_fh):
            # get sample from X containing historical and future exogenous data
            _x = self._get_x(_y, X, i)

            # forecast for self.horizon_length
            preds = self.forecaster.predict(y=_y, X=_x)

            # use predicted values as history for next iteration
            _y = self._concate_preds(_y, preds)

        # collect forecasting points from _y
        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = _y.index.get_level_values(-1).map(lambda x: x in absolute_horizons)
        preds = _y.loc[dateindex]

        return preds

    def _get_x(self, _y, X, i):
        if X is None:
            return None

        # keep till this index in X (hist from _y + future from _fh)
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
            from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT

            test_params = [
                {
                    "forecaster": PytorchForecastingTFT(
                        **PytorchForecastingTFT.get_test_params()[0],
                    ),
                    "horizon_length": 3,
                    "aggregate_method": None,
                }
            ]
        # TODO: use NaiveForecaster instead of PytorchForecastingTFT
        # as https://github.com/sktime/sktime/pull/6868/files is merged
        return test_params
