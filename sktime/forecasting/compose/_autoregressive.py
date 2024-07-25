# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""AutoRegressiveWrapper for Forecasters with fixed prediction length."""

from sktime.forecasting.base import _BaseGlobalForecaster


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
        "requires-fh-in-fit": True,  # TODO: look at this
        "capability:global_forecasting": True,
    }

    def __init__(self, forecaster, aggregate_method=None):
        self.forecaster = forecaster
        self.aggregate_method = (
            aggregate_method
            if aggregate_method is not None
            else (
                sum  # TODO: change the default aggregator
            )
        )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.forecaster.fit(y=y, X=X, fh=fh)

    def _predict(self, fh, X=None, y=None):
        pass

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
        # TODO: Fill this
        test_params = []
        params_no_broadcasting = [
            dict(p, **{"broadcasting": False}) for p in test_params
        ]
        test_params.extend(params_no_broadcasting)
        return test_params
