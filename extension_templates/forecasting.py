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

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from sktime.forecasting.base import BaseForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class MyForecaster(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
    _tags = {
        # tags and full specifications are available in the tag API reference
        # https://www.sktime.net/en/stable/api_reference/tags.html
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "univariate",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
        #
        # X-y-must-have-same-index = can estimator handle different X/y index?
        "X-y-must-have-same-index": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception if X.index is not contained in y.index
        #
        # enforce_index_type = index type that needs to be enforced in X/y
        "enforce_index_type": None,
        # valid values: pd.Index subtype, or list of pd.Index subtype
        # if not None, raises exception if X.index, y.index level -1 is not of that type
        #
        # handles-missing-data = can estimator handle missing data?
        "handles-missing-data": False,
        # valid values: boolean True (yes), False (no)
        # if False, raises exception if y or X passed contain missing data (nans)
        #
        # capability:insample = can forecaster make in-sample forecasts?
        "capability:insample": True,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if any forecast method called with in-sample fh
        #
        # capability:pred_int = does forecaster implement probabilistic forecasts?
        "capability:pred_int": False,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if proba methods are called (predict_interval etc)
        #
        # capability:pred_int:insample = can forecaster make in-sample proba forecasts?
        "capability:pred_int:insample": True,
        # valid values: boolean True (yes), False (no)
        # only needs to be set if capability:pred_int is True
        # if False, exception raised if proba methods are called with in-sample fh
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of th estimator
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party estimator, ensure to give credit to the
        # authors of the interfaced estimator
        #
        # maintainer = current maintainer(s) of the estimator
        # per algorithm maintainer role, see governance document
        # this is an "owner" type role, with rights and maintenance duties
        # for 3rd party interfaces, the scope is the sktime class only
        "maintainers": ["maintainer1", "maintainer2"],
        # valid values: str or list of str, should be GitHub handles
        # remove tag if maintained by sktime core team
        #
        # dependency tags: python version and soft dependencies
        # -----------------------------------------------------
        #
        # python version requirement
        "python_version": None,
        # valid values: str, PEP 440 valid python version specifiers
        # raises exception at construction if local python version is incompatible
        #
        # soft dependency requirement
        "python_dependencies": None,
        # valid values: str or list of str, PEP 440 valid package version specifiers
        # raises exception at construction if modules at strings cannot be imported
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.paramc etc
        # instead, write to self._paramc, self._newparam (starting with _)
        # example of handling conditional parameters or mutable defaults:
        if self.paramc is None:
            from sktime.somewhere import MyOtherEstimator

            self._paramc = MyOtherEstimator(foo=42)
        else:
            # estimators should be cloned to avoid side effects
            self._paramc = paramc.clone()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals should be used or fitted if needed
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
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

        # implement here
        # IMPORTANT: avoid side effects to X, fh

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
