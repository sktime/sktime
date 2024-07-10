# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for forecasters.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, cutoff, _fh,
    _cutoff, _converter_store_y, forecasters_, _tags, _tags_dynamic, _is_vectorized
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None)

Optional implements:
    updating                    - _update(self, y, X=None, update_params=True):
    predicting quantiles        - _predict_quantiles(self, fh, X=None, alpha=None)
    OR predicting intervals     - _predict_interval(self, fh, X=None, coverage=None)
    predicting variance         - _predict_var(self, fh, X=None, cov=False)
    distribution forecast       - _predict_proba(self, fh, X=None)
    fitted parameter inspection - _get_fitted_params()

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

__author__ = ["shlok191"]


import subprocess

import torch

from sktime.forecasting.base import BaseForecaster


class _LagLlamaAdapter(BaseForecaster):
    """Base class that interfaces the LagLlama forecaster.

    Parameters
    ----------
    pretrained : bool, optional (default=True)
        Dictates whether the pretrained LagLlama params should be initialized
    """

    _tags = {
        "y_inner_mtype": "gluonts_PandasDataset_panel",
        "X_inner_mtype": "gluonts_PandasDataset_panel",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": None,
        "python_dependencies": ["gluonts", "huggingface_hub", "lag_llama"],
    }

    def __init__(
        self,
        device=None,
        context_length=None,
        prediction_length=None,
        num_samples=None,
        batch_size=None,
        nonnegative_pred_samples=None,
        lr=None,
        trainer_kwargs=None,
    ):
        from lag_llama.gluon.estimator import LagLlamaEstimator

        # Defining private variable values
        self.device = device
        self.device_ = torch.device("cpu") if not device else device

        self.context_length = context_length
        self.context_length_ = 32 if not context_length else context_length

        self.prediction_length = prediction_length
        self.prediction_length_ = 100 if not prediction_length else prediction_length

        self.num_samples = num_samples
        self.num_samples_ = 10 if not num_samples else num_samples

        self.batch_size = batch_size
        self.batch_size_ = 32 if not batch_size else batch_size

        # Now storing the training related variables
        self.lr = lr
        self.lr_ = 5e-5 if not lr else lr

        self.trainer_kwargs = trainer_kwargs
        self.trainer_kwargs_ = (
            {"max_epochs": 50} if not trainer_kwargs else trainer_kwargs
        )

        # Not storing private variables for boolean specific values
        self.nonnegative_pred_samples = nonnegative_pred_samples

        super().__init__()

        # Downloading the LagLlama weights from Hugging Face
        download_command = (
            "huggingface-cli download time-series-foundation"
            + "-models/Lag-Llama lag-llama.ckpt --local-dir ."
        )

        status = subprocess.run(
            download_command, shell=True, check=True, capture_output=True
        )

        # Checking if the command ran successfully
        if status.returncode != 0:
            raise RuntimeError(
                "Failed to fetch the pretrained model weights from HuggingFace!"
            )

        # Load in the lag llama checkpoint
        ckpt = torch.load("./lag-llama.ckpt", map_location=self.device_)

        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        self.estimator_args = estimator_args

        # By default, we maintain RoPE scaling
        # We provide the user an option to disable in fit() function
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(
                1.0,
                (self.context_length_ + self.prediction_length_)
                / estimator_args["context_length"],
            ),
        }

        # Creating our LagLlama estimator
        self.estimator_ = LagLlamaEstimator(
            ckpt_path="lag-llama.ckpt",
            prediction_length=self.prediction_length_,
            context_length=self.context_length_,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=self.batch_size_,
            device=self.device_,
            rope_scaling=rope_scaling_arguments,
        )

        lightning_module = self.estimator_.create_lightning_module()
        transformation = self.estimator_.create_transformation()

        # Finally, we create our predictor!
        self.predictor_ = self.estimator_.create_predictor(
            transformation, lightning_module
        )

        # Since we are importing pretrained weights
        self._is_fitted = True

    # todo: implement this, mandatory
    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : GluonTS PandasDataset Object, optional (default=None)
            Time series to which to fit the forecaster.

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.

        X : GluonTS PandasDataset Object, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        from lag_llama.gluon.estimator import LagLlamaEstimator

        # Creating a new LagLlama estimator with the appropriate
        # forecasting horizon
        self.estimator_ = LagLlamaEstimator(
            ckpt_path="lag-llama.ckpt",
            prediction_length=32 if not fh else fh,  # This is the most important here!
            context_length=self.context_length_,
            input_size=self.estimator_args["input_size"],
            n_layer=self.estimator_args["n_layer"],
            n_embd_per_head=self.estimator_args["n_embd_per_head"],
            n_head=self.estimator_args["n_head"],
            scaling=self.estimator_args["scaling"],
            time_feat=self.estimator_args["time_feat"],
            batch_size=self.batch_size_,
            device=self.device_,
            lr=self.lr_,
            trainer_kwargs=self.trainer_kwargs_,
        )

        lightning_module = self.estimator.create_lightning_module()
        transformation = self.estimator.create_transformation()

        # Creating a new predictor
        self.predictor_ = self.estimator_.create_predictor(
            transformation, lightning_module
        )

        # Lastly, training the model
        self.predictor_ = self.estimator.train(y.train, cache_data=True)

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
            The forecasting horizon is not read here, please set it as
            `prediction_length` during initialization or when calling the fit function

        X : GluonTS PandasDataset Object (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : GluonTS PandasDataset Object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        from gluonts.evaluation import make_evaluation_predictions

        # Creating a forecaster object
        forecast_it, _ = make_evaluation_predictions(
            dataset=X, predictor=self.predictor_, num_samples=self.num_samples_
        )

        # Forming a list of the forecasting iterations
        forecasts = list(forecast_it)

        return forecasts

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
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

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update_predict_single method
    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X)
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: consider implementing one of _predict_quantiles and _predict_interval
    #   if one is implemented, the other one works automatically
    #   when interfacing or implementing, consider which of the two is easier
    #   both can be implemented if desired, but usually that is not necessary
    #
    # if _predict_var or _predict_proba is implemented, this will have a default
    #   implementation which uses _predict_proba or _predict_var under normal assumption
    #
    # if implementing _predict_interval, delete _predict_quantiles
    # if not implementing either, delete both methods
    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh, alpha
        #
        # Note: unlike in predict_quantiles where alpha can be float or list of float
        #   alpha in _predict_quantiles is guaranteed to be a list of float

    # implement one of _predict_interval or _predict_quantiles (above), or delete both
    #
    # if implementing _predict_quantiles, delete _predict_interval
    # if not implementing either, delete both methods
    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh, coverage
        #
        # Note: unlike in predict_interval where coverage can be float or list of float
        #   coverage in _predict_interval is guaranteed to be a list of float

    # todo: consider implementing _predict_var
    #
    # if _predict_proba or interval/quantiles are implemented, this will have a default
    #   implementation which uses _predict_proba or quantiles under normal assumption
    #
    # if not implementing, delete _predict_var
    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        # implement here
        # implementing the cov=True case is optional and can be omitted

    # todo: consider implementing _predict_proba
    #
    # if interval/quantiles or _predict_var are implemented, this will have a default
    #   implementation which uses variance or quantiles under normal assumption
    #
    # if not implementing, delete _predict_proba
    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
            if has not been passed in fit, must be passed, not optional
        X : sktime time series object, optional (default=None)
                Exogeneous time series for the forecast
            Should be of same scitype (Series, Panel, or Hierarchical) as y in fit
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain fh.index and y.index both
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        # implement here
        # returned BaseDistribution should have same index and columns
        # as the predict return
        #
        # implementing the marginal=False case is optional and can be omitted

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def _predict_moving_cutoff(self, y, cv, X=None, update_params=True):
        """Make single-step or multi-step moving cutoff predictions.

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator
        X : pd.DataFrame
        update_params : bool

        Returns
        -------
        y_pred = pd.Series
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, cv

    # todo: consider implementing this, optional
    # implement only if different from default:
    #   default retrieves all self attributes ending in "_"
    #   and returns them with keys that have the "_" removed
    # if not implementing, delete the method
    #   avoid overriding get_fitted_params
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
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
