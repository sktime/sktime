# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""AutoRegressiveWrapper for Forecasters with fixed prediction length."""

__author__ = ["geetu040"]

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
        "ignores-exogeneous-X": True,  # TODO: look at this
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, forecaster, horizon_length, aggregate_method=None):
        self.forecaster = forecaster
        self.horizon_length = horizon_length
        # TODO: check the need for this
        self.aggregate_method = (
            aggregate_method
            if aggregate_method is not None
            else (
                sum  # TODO: change the default aggregator
            )
        )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        # ignore the fh provided fit
        _fh = ForecastingHorizon(range(1, self.horizon_length))

        self.forecaster.fit(y=y, X=X, fh=_fh)

    def _predict(self, fh, X=None, y=None):
        # use fh to find the maximum length to forecast regressively
        max_fh = max(
            *(fh.to_relative(self._cutoff)._values + 1),
            self.horizon_length,
        )

        # minimum number of iterations needed to exceed forecast till max_fh
        n_iterations = (max_fh + self.horizon_length - 1) // self.horizon_length

        # regressive forecasting loop
        y_ = self._y if y is None else y
        for _ in range(n_iterations):
            preds = self.forecaster.predict(y=y_)
            y_ = pd.concat([y_, preds])
            # TODO: update this concatenation for multi-index

        # collect forecasting points from y_
        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = y_.index.get_level_values(-1).map(lambda x: x in absolute_horizons)
        preds = y_.loc[dateindex]

        return preds

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
                        trainer_params={
                            "max_epochs": 1,
                            "limit_train_batches": 10,
                        },
                        dataset_params={
                            "max_encoder_length": 3,
                        },
                        random_log_path=True,
                    ),
                    "horizon_length": 3,
                }
            ]
        return test_params
