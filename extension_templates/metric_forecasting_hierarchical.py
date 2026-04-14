"""Extension template for hierarchical forecasting performance metrics.

Purpose of this implementation template:
    quick implementation of new hierarchical forecasting metrics following template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new hierarchical metric:
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
    metric evaluation - _evaluate(self, y_true, y_pred, **kwargs)
        OR _evaluate_by_index(self, y_true, y_pred, **kwargs)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

# todo: add any necessary imports here


class MyHierarchicalForecastingMetric(BaseForecastingErrorMetric):
    """Custom hierarchical forecasting performance metric.

    todo: write docstring, describing your custom hierarchical forecasting metric

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
        # tags and full specifications are available in tag API reference
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
        # learning_type = learning type of metric
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
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel or Hierarchical
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, y_true/y_pred are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # inner_implements_multilevel = whether metric implements hierarchical logic
        "inner_implements_multilevel": True,
        # valid values: True = metric handles hierarchical data directly
        #   False = metric operates on flattened data
        #
        # metric_type = type of metric (loss vs score)
        "metric_type": "loss",
        # valid values: "loss" (lower is better), "score" (higher is better)
        #
        # metric_scope = scope of metric evaluation
        "metric_scope": "pointwise",
        # valid values: "pointwise", "distributional", "interval"
        #
        # requires-y-train = whether metric requires training data
        "requires-y-train": False,
        # valid values: True = metric requires y_train parameter
        #   False = metric does not require training data
        #
        # requires-y-pred-benchmark = whether metric requires benchmark predictions
        "requires-y-pred-benchmark": False,
        # valid values: True = metric requires y_pred_benchmark parameter
        #   False = metric does not require benchmark predictions
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of metric
        # an author is anyone with significant contribution to code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party metric, ensure to give credit to the
        # authors of the interfaced metric
        #
        # maintainer = current maintainer(s) of metric
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

    # todo: implement this, mandatory - choose either _evaluate or _evaluate_by_index
    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate hierarchical forecasting metric.

        core logic

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.
            y can be a pd.DataFrame in one of following formats:

            * pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            * pd.DataFrame with row MultiIndex, last level is time index
              (hierarchical structure)

        y_pred : pandas.DataFrame
            Predicted values to evaluate.
            y can be a pd.DataFrame in one of following formats:

            * pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            * pd.DataFrame with row MultiIndex, last level is time index
              (hierarchical structure)

        **kwargs : dict
            Additional keyword arguments that may be required by specific metrics.
            Common arguments include:
            - sample_weight : array-like or callable, default=None
                Sample weights for each time point.
            - y_train : pandas.DataFrame, default=None
                Training data used to normalize error metric.
            - y_pred_benchmark : pandas.DataFrame, default=None
                Benchmark predictions to compare y_pred to.

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.
            Hierarchical structure is preserved if multilevel="raw_values".

            * float if multioutput="uniform_average" or array-like,
              and multilevel="uniform_average" or "uniform_average_time".
              Value is metric averaged over variables and levels.
            * np.ndarray of shape (y_true.columns,) 
              if multioutput="raw_values"
              and multilevel="uniform_average" or "uniform_average_time".
              i-th entry is metric calculated for i-th variable
            * pd.DataFrame if multilevel="raw_values".
              of shape (n_levels, ) if multioutput="uniform_average";
              of shape (n_levels, y_true.columns) if multioutput="raw_values".
              metric is applied per level, row averaging as in multioutput.
        """

        # implement here
        # IMPORTANT: avoid side effects to y_true, y_pred
        
        # Example implementation for hierarchical metric (replace with your actual metric logic):
        import numpy as np
        import pandas as pd
        
        # Example: hierarchical mean absolute error with level-specific weighting
        # 1. Calculate absolute errors at each level
        absolute_errors = np.abs(y_true - y_pred)
        
        # 2. Handle hierarchical structure
        if isinstance(y_true.index, pd.MultiIndex):
            # Hierarchical data - calculate metric per level
            level_names = y_true.index.names[:-1]  # All levels except time
            time_level = y_true.index.names[-1]   # Time level
            
            # Group by hierarchy levels and calculate metric
            if self.multioutput == "raw_values":
                # Return metric per level and per variable
                level_metrics = {}
                for level in level_names:
                    level_mask = y_true.index.get_level_values(level)
                    level_errors = absolute_errors.groupby(level_mask).mean()
                    level_metrics[level] = level_errors
                
                return pd.DataFrame(level_metrics)
            else:
                # Average across levels
                overall_metric = absolute_errors.mean().mean()
                return overall_metric
        else:
            # Non-hierarchical data - use standard metric calculation
            if self.multioutput == "raw_values":
                return absolute_errors.mean()
            else:
                return absolute_errors.mean().mean()

    # todo: alternatively, implement this instead of _evaluate (for time-specific metrics)
    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return hierarchical metric evaluated at each time point.

        core logic

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.
            Can be hierarchical with MultiIndex.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.
            Can be hierarchical with MultiIndex.

        **kwargs : dict
            Additional keyword arguments passed to metric function.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.
            Hierarchical structure is preserved if multilevel="raw_values".

            * pd.Series if multioutput="uniform_average" or array-like.
              index is equal to index of y_true (time level);
              entry at index i is metric at time i, averaged over variables
            * pd.DataFrame if multioutput="raw_values".
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j
        """
        
        # Example implementation - replace with your actual time-specific metric logic
        import numpy as np
        import pandas as pd
        
        # Calculate absolute errors at each time point
        absolute_errors = np.abs(y_true - y_pred)
        
        if isinstance(y_true.index, pd.MultiIndex):
            # Hierarchical data
            time_index = y_true.index.get_level_values(-1)  # Time level
            
            if self.multioutput == "raw_values":
                # Return metric per time point and per variable
                result_df = pd.DataFrame(
                    index=time_index.unique(),
                    columns=y_true.columns,
                    dtype="float64"
                )
                
                for time_point in time_index.unique():
                    time_mask = y_true.index.get_level_values(-1) == time_point
                    time_errors = absolute_errors[time_mask]
                    result_df.loc[time_point] = time_errors.mean(axis=0)  # Average over hierarchy levels
                
                return result_df
            else:
                # Average across variables at each time point
                result_series = pd.Series(
                    index=time_index.unique(),
                    dtype="float64"
                )
                
                for time_point in time_index.unique():
                    time_mask = y_true.index.get_level_values(-1) == time_point
                    time_errors = absolute_errors[time_mask]
                    result_series.loc[time_point] = time_errors.mean()  # Average over hierarchy and variables
                
                return result_series
        else:
            # Non-hierarchical data
            if self.multioutput == "raw_values":
                return absolute_errors.mean(axis=1)
            else:
                return absolute_errors.mean(axis=1).mean()

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for hierarchical metrics.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set testing parameters for hierarchical metrics
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
        #       "big" metrics which not only increase test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum of two such parameter sets with different
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
        # Default test parameters for hierarchical forecasting metric
        params = {
            "parama": 1, 
            "paramb": "default", 
            "paramc": True,
            "multioutput": "uniform_average",
            "multilevel": "uniform_average"
        }
        return params
