"""Extension template for forecasting metrics (non-hierarchical).

Purpose of this implementation template:
    Quick implementation of new estimators following the template.
    This is not a concrete class or Base class to import!
    Use this as a starting template to build on.

    How to use:
    - Copy the template to a suitable folder and give it a descriptive name.
    - Work through all the todo comments given.
    - Ensure to implement the mandatory methods.
    - Do not write in reserved variables: _tags.
    - You can add more private methods, but do not override BaseEstimator's
      private methods. An easy way to be safe is to prefix your methods with
      "_custom".
    - Change docstrings for functions and the file.
    - Ensure interface compatibility by testing
      performance_metrics/forecasting/tests
    - Once complete: use as a local library, or contribute to sktime via PR.
    - More details:
      https://www.sktime.net/en/stable/developer_guide/add_estimators.html

    Mandatory methods to implement (at least one of):
        evaluating at each time index  -
            _evaluate_by_index(self, y_true, y_pred, **kwargs)
        overall evaluation             -
            _evaluate(self, y_true, y_pred, **kwargs)

    Note: if only ``_evaluate_by_index`` is implemented, ``_evaluate``
    defaults to the arithmetic mean of ``_evaluate_by_index`` over time points.
    If only ``_evaluate`` is implemented, ``_evaluate_by_index`` defaults to
    jackknife pseudosamples from ``_evaluate``.
    Optimally, both are implemented for best performance and correctness.

    Testing - required for sktime test framework and check_estimator usage:
        get default parameters for test instance(s) - get_test_params()

    copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top


# todo: change class name and write docstring
class MyForecastingMetric(BaseForecastingErrorMetric):
    """Custom forecasting metric. todo: write docstring.

    todo: describe your custom forecasting metric here.

    This template is for non-hierarchical forecasting metrics, i.e., metrics
    that do not implement native handling of hierarchical (multi-level) data.
    For hierarchical metrics, use the ``metric_forecasting_hierarchical``
    template instead.

    At least one of ``_evaluate`` or ``_evaluate_by_index`` must be implemented.
    If only ``_evaluate_by_index`` is implemented, ``_evaluate`` defaults to
    the arithmetic mean over time points. Optimally, implement both.

    Parameters
    ----------
    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), returns per-variable errors.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call averages over time points, equivalent to ``evaluate``.
        * If ``True``, direct call evaluates at each time point,
          equivalent to ``evaluate_by_index``.

    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    """

    # optional todo: override base class estimator default tags here if necessary
    # these are the default values - only change if your metric differs
    _tags = {
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        #     if interfacing a 3rd party estimator, ensure to give credit to the
        #     authors of the interfaced estimator
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        #     for 3rd party interfaces, the scope is the sktime class only
        # specify one or multiple authors and maintainers, only for sktime contribution
        # remove maintainer tag if maintained by sktime core team
        #
        # estimator tags
        # --------------
        "object_type": ["metric_forecasting", "metric"],
        # the prediction type this metric is designed for:
        # "pred" = point forecasts (default)
        # "pred_interval" = prediction intervals
        # "pred_quantiles" = quantile forecasts
        # "pred_var" = variance forecasts
        "scitype:y_pred": "pred",
        # whether the metric requires training data y_train to be passed
        "requires-y-train": False,
        # whether the metric requires a benchmark forecast y_pred_benchmark
        "requires-y-pred-benchmark": False,
        # whether the metric supports multivariate (multi-column) inputs
        "capability:multivariate": True,
        # whether lower values of the metric are better (True) or worse (False)
        "lower_is_better": True,
        # set to True only if _evaluate natively handles hierarchical (multilevel)
        # data with pd.MultiIndex. If False (default), the framework handles this
        # via vectorization over hierarchy levels automatically.
        "inner_implements_multilevel": False,
    }

    # todo: add hyper-parameters to constructor, keep multioutput/multilevel/by_index
    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        parama=None,
        paramb="default",
    ):
        # todo: write any hyper-parameters to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: do not overwrite or mutate self.params from here on
        # for handling defaults etc, write to other attributes, e.g., self._parama

        # leave this as is
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Mandatory to implement if ``_evaluate`` is not implemented.
        If implemented, ``_evaluate`` defaults to the mean of this over time.

        Parameters
        ----------
        y_true : pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate against ground truth.
            Same format as ``y_true``, same indices and columns.

        **kwargs : dict, optional
            Optional keyword arguments, may include:

            y_train : pd.DataFrame, same columns as y_true
                Training data, required if ``requires-y-train`` tag is True.
            y_pred_benchmark : pd.DataFrame, same format as y_pred
                Benchmark predictions, required if tag
                ``requires-y-pred-benchmark`` is True.
            sample_weight : 1D array-like of length len(y_true), default=None
                Per-time-point sample weights.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Metric evaluated at each time point.

            * ``pd.Series`` if ``multioutput="uniform_average"`` or array-like.
              Index equals index of ``y_true``.
              Entry at index i is the metric at time i, averaged over variables.
            * ``pd.DataFrame`` if ``multioutput="raw_values"``.
              Index and columns equal those of ``y_true``.
              Entry (i, j) is the metric at time i for variable j.
        """
        raise NotImplementedError("abstract method")
        # todo: implement your metric logic here
        # The example below implements mean absolute error at each time point.
        # Replace with your own logic.
        #
        # multioutput = self.multioutput
        #
        # raw_values = (y_true - y_pred).abs()
        #
        # # apply sample weights if provided (use the helper method)
        # raw_values = self._get_weighted_df(raw_values, **kwargs)
        #
        # # handle multioutput averaging (use the helper method)
        # return self._handle_multioutput(raw_values, multioutput)

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric overall (aggregated over time points).

        Optional to implement. If not implemented, defaults to the arithmetic
        mean of ``_evaluate_by_index`` over time points.

        Override this method for efficiency, or if your aggregation is not
        a simple mean over time points (e.g., a weighted or non-linear
        aggregation such as MASE, MAPE, or geometric mean).

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth target values.
        y_pred : pd.DataFrame
            Predicted values to evaluate.
        **kwargs : dict, optional
            Same optional kwargs as ``_evaluate_by_index``.

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.

            * float if ``multioutput="uniform_average"`` or array-like.
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if ``multioutput="raw_values"``.
        """
        # default: mean over time points from _evaluate_by_index
        # override this for non-simple aggregations or performance
        index_df = self._evaluate_by_index(y_true, y_pred, **kwargs)
        return index_df.mean(axis=0)

    # todo: return default parameters so that a test instance can be created
    # required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be a dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #
        # A good parameter set should primarily satisfy two criteria:
        #   1. Low testing time - ideally a few seconds for the entire test suite.
        #   2. At least two parameter sets with different values for good coverage.
        #
        # example 1: single parameter dict
        # params = {"parama": 1, "paramb": "option1"}
        #
        # example 2: list of parameter dicts
        # params = [{"parama": 1, "paramb": "option1"},
        #           {"parama": 2, "paramb": "option2"}]

        # todo: replace with your actual test parameters
        params1 = {"parama": 1, "paramb": "default"}
        params2 = {"parama": 2, "paramb": "option2"}
        return [params1, params2]
