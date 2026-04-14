"""Extension template for forecasting performance metrics.

Purpose of this implementation template:
    quick implementation of new forecasting metrics following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new metric:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory methods to implement:
    metric evaluation - evaluate(y_true, y_pred, **kwargs)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric as BaseForecastingMetric

# todo: add any necessary imports here


class MyForecastingMetric(BaseForecastingMetric):
    """Custom forecasting performance metric.

    todo: write docstring, describing your custom forecasting metric

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=True)
        descriptive explanation of paramc
    and so on
    """

    _tags = {
        # tags and full specifications are available in the tag API reference
        # https://www.sktime.net/en/stable/api_reference/tags.html
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="metric", as_dataframe=True)
        #
        # estimator tags
        # --------------
        #
        # task type - type of task this metric is designed for
        "task_type": "forecasting",
        # valid values: "forecasting", "classification", "regression", "clustering"
        #
        # learning_type = learning type of the metric
        "learning_type": "metric",
        # valid values: "metric", "loss"
        #
        # capability:multivariate controls whether metric can handle multivariate series
        "capability:multivariate": True,
        # valid values: True = metric can handle multivariate series
        #   False = metric only handles univariate series
        #
        # capability:missing_data = can metric handle missing data?
        "capability:missing_data": False,
        # valid values: boolean True (yes), False (no)
        # if False, raises exception if y_true or y_pred contain missing data (nans)
        #
        # y_inner_mtype control which format y_true/y_pred appear in evaluate()
        "y_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series or Panel
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, y_true/y_pred are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # metric_type = type of metric (loss vs score)
        "metric_type": "loss",
        # valid values: "loss" (lower is better), "score" (higher is better)
        #
        # metric_scope = scope of metric evaluation
        "metric_scope": "pointwise",
        # valid values: "pointwise", "distributional", "interval"
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of the metric
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party metric, ensure to give credit to the
        # authors of the interfaced metric
        #
        # maintainer = current maintainer(s) of the metric
        # per algorithm maintainer role, see governance document
        # this is an "owner" type role, with rights and maintenance duties
        # for 3rd party interfaces, scope is the sktime class only
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

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=True):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.paramc etc
        # instead, write to self._paramc, self._newparam (starting with _)
        # example of handling conditional parameters or mutable defaults:
        if not self.paramc:
            self._paramc = False
        else:
            self._paramc = True

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if self.parama > 10:
        #   self.set_tags(capability:multivariate=False)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "capability:missing_values"])

    # todo: implement this, mandatory
    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the forecasting metric.

        core logic

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
                 where fh is forecasting horizon
            Ground truth (correct) target values.
        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
                 where fh is forecasting horizon
            Forecasted values.
        **kwargs : dict
            Additional keyword arguments that may be required by specific metrics.
            Common arguments include:
            - horizon_weight : array-like of shape (fh,), default=None
                Forecast horizon weights.
            - multioutput : {'raw_values', 'uniform_average'} or array-like of shape 
                (n_outputs,), default='uniform_average'
                Defines how to aggregate metric for multivariate (multioutput) data.
            - y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or 
                (n_timepoints, n_outputs), default = None
                Observed training values (required for scaled metrics).
            - sp : int, default=1
                Seasonal periodicity of training data (required for scaled metrics).

        Returns
        -------
        metric_value : float or array
            Computed metric value.
            If multioutput is 'raw_values', then metric is returned for each
            output separately.
            If multioutput is 'uniform_average' or an ndarray of weights, then
            weighted average metric of all output errors is returned.
        """

        # implement here
        # IMPORTANT: avoid side effects to y_true, y_pred
        
        # Example implementation (replace with your actual metric logic):
        # 1. Input validation and preprocessing
        # 2. Compute metric-specific calculations
        # 3. Handle multivariate aggregation if needed
        # 4. Return final metric value
        
        # Placeholder implementation - replace with your metric
        import numpy as np
        
        # Convert to numpy arrays for computation
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Example: simple mean absolute error
        absolute_errors = np.abs(y_true - y_pred)
        
        # Handle horizon weights if provided
        horizon_weight = kwargs.get("horizon_weight", None)
        if horizon_weight is not None:
            horizon_weight = np.asarray(horizon_weight)
            if len(horizon_weight) != len(absolute_errors):
                raise ValueError("horizon_weight must have same length as predictions")
            metric_value = np.average(absolute_errors, weights=horizon_weight)
        else:
            metric_value = np.mean(absolute_errors)
        
        # Handle multioutput aggregation
        multioutput = kwargs.get("multioutput", "uniform_average")
        if multioutput == "raw_values":
            return metric_value
        else:
            return np.mean(metric_value)

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for metrics.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for metrics
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as metrics from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at top
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
        #       This is vital for cases where default values result in
        #       "big" metrics which not only increases test time but also
        #       run into risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"parama": value0, "paramb": value1, "paramc": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"parama": value1, "paramb": value2},
        #           {"parama": value3, "paramb": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"parama": value1, "paramb": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"parama": value3, "paramb": value4}
        # return params
        #
        # Default test parameters for forecasting metric
        params = {"parama": 1, "paramb": "default", "paramc": True}
        return params
