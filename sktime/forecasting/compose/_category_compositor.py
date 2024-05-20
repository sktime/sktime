# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a compositor to utilize forecasters based on ADI/CV categorization."""

__author__ = ["shlok191"]

from sktime.forecasting.base import BaseForecaster
from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.adi_cv import ADICVTransformer


class CategoryCompositor(BaseForecaster):
    """Compositor that utilizes varying forecasters by time series data's nature.

    Applies a series-to-primitives transformer on a given time series and utilizes
    generated primitive value to apply the most appropriate forecaster to the
    given series.

    Parameters
    ----------
    forecasters : dict[sktime forecasters]
        dict of forecasters with the key corresponding to categories generated
        by the given transformer and the value corresponding to a sktime forecaster.

    transformer : sktime transformer, default = ADICVTransformer()
        A series-to-primitives sk-time transformer that generates a value
        which can be used to quantify a choice of forecaster for the time series.

    Raises
    ------
    AssertionError: If a valid transformer (an instance of BaseTransformer)
    is not passed or if valid forecasters (instances of BaseForecaster) is not given.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:pred_int": False,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": None,
        "python_dependencies": [
            "BaseForecaster",
            "BaseTransformer" "ADICVTransformer",
        ],
    }

    def __init__(self, forecasters, transformer=ADICVTransformer):
        # saving arguments to object storage
        self.transformer = transformer
        self.forecasters = forecasters

        super().__init__()

        # validating passed arguments
        assert isinstance(transformer, BaseTransformer)

        for forecaster in forecasters.values():
            assert isinstance(forecaster, BaseForecaster)

        # All checks OK!

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : Pd.Series
            The target time series to which we fit the data.

        fh : ForecastingHorizon | None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.

        X : Pd.Series | None, optional (default=None)
            No exogenous variables are used for this.

        Returns
        -------
        self : reference to self
        """
        # reset the state of the forecaster
        self.reset()

        # asserting validity of input
        assert y is not None

        # passing time series through the provided transformer!
        self.category_ = self.transformer.fit_transform(y=y).iloc[0, 0]

        # check if we have an available forecaster
        assert (
            self.category_ in self.forecasters
        ), f"Forecaster not provided for given time series of type {self.category_}."

        self.chosen_forecaster_ = self.forecasters[self.category_]
        self.chosen_forecaster_.fit(y=y, X=X, fh=fh)  # fitting the forecaster!

        self._is_fitted = True

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here

        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        # Check if we have a fited forecaster to the data
        self.check_is_fitted()

        # Obtain the prediction values for the given horizon.
        y_pred = self.chosen_forecaster_(fh=fh, X=X)

        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        # Check if we have a a fitted model and update it if we do
        self.check_is_fitted()
        self.chosen_forecaster_.update(y=y, X=X, update_params=update_params)

    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X)

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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params
