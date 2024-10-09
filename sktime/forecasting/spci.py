# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SPCI Forecaster."""

_all_ = ["SPCIForecaster"]
__author__ = ["ksharma6"]

from sklearn.utils import check_random_state
from skpro.regression.base import BaseProbaRegressor

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class SPCI(BaseForecaster):
    """Sequential Predictive Conformal Inference Forecaster.

    SPCI is a model-free and distribution-free framework that combines
    a Sktime forecaster with a quantile regression model to perform Conformal
    Predictions on time series data [1].

    The algorithm works like so:
    1. Obtain point predictions, ``y_preds``, and point prediction residuals,
    ``e^= y - y_preds``, using Sktime forecaster.
    2. For t > T do`:
        3. Fit quantile regressor onto ``e^``
        4. Use quantile regression to obtain quantile predictions, ``q_pred``
        5. Calculate prediction interval at time ``t``
        6. Calculate the new residual ``e^_t``
        7. Update residuals ``e^`` by sliding one index forward
        (i.e. add ``e^_t`` and remove the oldest one)

    returns: Prediction intervals

    Parameters
    ----------
    forecaster : estimator object
        The base forecaster to fit in order to make point predictions.
    regressor: skpro regressor object
        The base regressor to fit in order to predict quantiles.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.


    Examples
    --------
    continue - return once implementation is completed.

    References
    ----------
    .. [1] Chen Xu & Yao Xie (2023). Sequential Predictive
    Conformal Inference for Time Series.

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
        "authors": ["ksharma6"],
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
        "maintainers": ["ksharma6"],
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
    def __init__(
        self,
        forecaster=None,
        regressor_proba=None,
        random_state=None,
    ):
        self.forecaster = forecaster
        self.forecaster_ = (
            forecaster.clone() if forecaster is not None else NaiveForecaster()
        )
        self.regressor_proba = regressor_proba
        self.regressor_proba_ = (
            regressor_proba.clone()
            if regressor_proba is not None
            else BaseProbaRegressor()
        )
        self.random_state = random_state

        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

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
        """Calculate point predictions and fit regressor to point predictions.

        Parameters
        ----------
        y : sktime compatible tabular data container, Table scitype
            numpy1D iterable, of shape [n_instances]
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
             pd-multiindex: pd.DataFrame with columns = variables,
             index = pd.MultiIndex with first level = instance indices,
             second level = time indices
             numpy3D: 3D np.array (any number of dimensions, equal length series)
             of shape [n_instances, n_dimensions, series_length]
             or of any other supported Panel mtype
             for list of mtypes, see datatypes.SCITYPE_REGISTER
             for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Raises
        ------
        ValueError
            If fh is not equal to 1.

        Returns
        -------
        self : reference to self
        """
        self._fh = fh

        # random state handling passed into input estimators
        self.random_state_ = check_random_state(self.random_state)

        # calculate + store point predictions from forecaster_
        f_hats = []

        for i in range(len(y) - fh):
            self.forecaster_.fit(y=y[: i + fh], X=X[: i + fh], fh=fh)
            f_hat = self.forecaster_.predict(fh=fh, X=X)
            f_hats.append(f_hat)

        # calculate + store residuals from regressor_proba_
        residuals = []
        for i in range(len(f_hats)):
            residual = f_hats[i] - y[i]
            residuals.append(residual)

        # fit regressor to point prediction residuals
        self.regressor_proba_.fit(X=residuals, y=y)

        return self

    def _predict(self, X):
        """Calculate upper and lower bounds of prediction intervals.

        Parameters
        ----------
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
             pd-multiindex: pd.DataFrame with columns = variables,
             index = pd.MultiIndex with first level = instance indices,
             second level = time indices
             numpy3D: 3D np.array (any number of dimensions, equal length series)
             of shape [n_instances, n_dimensions, series_length]
             or of any other supported Panel mtype
             for list of mtypes, see datatypes.SCITYPE_REGISTER
             for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : sktime compatible tabular data container, Table scitype
            numpy1D iterable, of shape [n_instances]
        """
        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    def _update(self, y, X, update_params=True):
        """Update time series to increment training data.

        Parameters
        ----------
        y : sktime compatible tabular data container, Table scitype
            numpy1D iterable, of shape [n_instances]
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
             pd-multiindex: pd.DataFrame with columns = variables,
             index = pd.MultiIndex with first level = instance indices,
             second level = time indices
             numpy3D: 3D np.array (any number of dimensions, equal length series)
             of shape [n_instances, n_dimensions, series_length]
             or of any other supported Panel mtype
             for list of mtypes, see datatypes.SCITYPE_REGISTER
             for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        self : reference to self
        """
        self.fit(y=self._y, X=self._X, fh=self._fh)
        return self

    def _predict_interval(self, X, coverage):
        """Compute/return prediction interval forecasts.

        Parameters
        ----------
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
             pd-multiindex: pd.DataFrame with columns = variables,
             index = pd.MultiIndex with first level = instance indices,
             second level = time indices
             numpy3D: 3D np.array (any number of dimensions, equal length series)
             of shape [n_instances, n_dimensions, series_length]
             or of any other supported Panel mtype
             for list of mtypes, see datatypes.SCITYPE_REGISTER
             for specifications, see examples/AA_datatypes_and_datasets.ipynb
        coverage : float or list, optional (default=0.95)
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
        # logic will contain _predict_residuals(forecasts_preds)

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
