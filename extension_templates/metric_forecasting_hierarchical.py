"""Extension template for forecasting metrics (hierarchical).

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

    This template is for HIERARCHICAL forecasting metrics, i.e., metrics that
    natively handle hierarchical (multi-level) data with pd.MultiIndex, and
    implement per-level output when ``multilevel="raw_values"``.

    Use this template when your metric aggregates differently across hierarchy
    levels than a simple mean, requires access to the full hierarchy structure,
    or needs to return per-level results natively via ``multilevel="raw_values"``.

    For non-hierarchical metrics, use the ``metric_forecasting`` template.

    Mandatory methods to implement (at least one of):
        overall evaluation             -
            _evaluate(self, y_true, y_pred, **kwargs)
        evaluating at each time index  -
            _evaluate_by_index(self, y_true, y_pred, **kwargs)

    Note: if only ``_evaluate`` is implemented, ``_evaluate_by_index`` defaults
    to jackknife pseudosamples from ``_evaluate``.
    If only ``_evaluate_by_index`` is implemented, ``_evaluate`` defaults to
    the arithmetic mean over time points.
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
class MyForecastingMetricHierarchical(BaseForecastingErrorMetric):
    """Custom hierarchical forecasting metric. todo: write docstring.

    todo: describe your custom forecasting metric here.

    This template is for HIERARCHICAL forecasting metrics, i.e., metrics that
    natively handle hierarchical (multi-level) data with pd.MultiIndex, and
    implement per-level output when ``multilevel="raw_values"``.

    The key difference from the non-hierarchical template:
    - Tag ``inner_implements_multilevel`` is set to ``True``.
    - ``_evaluate`` receives a ``pd.DataFrame`` with a row ``MultiIndex``
      where the last level is the time index and preceding levels are hierarchy.
    - ``_evaluate`` should return a ``pd.DataFrame`` of shape
      ``(n_levels, n_columns)`` when ``multilevel="raw_values"``,
      or a float/array when ``multilevel`` leads to averaging.

    At least one of ``_evaluate`` or ``_evaluate_by_index`` must be implemented.
    If only ``_evaluate`` is implemented, ``_evaluate_by_index`` defaults to
    jackknife pseudosamples. Optimally, implement both.

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
          does not average errors across levels - per-level results are returned.
          Returns ``pd.DataFrame`` of shape ``(n_levels, n_columns)``,
          or ``(n_levels,)`` if ``multioutput`` leads to averaging.

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
        # IMPORTANT: set to True for hierarchical templates.
        # Tells the framework that _evaluate natively handles hierarchical data
        # with pd.MultiIndex; framework will NOT vectorize over hierarchy levels.
        # Your _evaluate must handle the MultiIndex pd.DataFrame directly.
        "inner_implements_multilevel": True,
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

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric on hierarchical inputs.

        Mandatory to implement if ``_evaluate_by_index`` is not implemented.
        Must handle hierarchical data natively (``inner_implements_multilevel=True``).

        If not implemented, defaults to the mean of ``_evaluate_by_index``
        over time points (via jackknife pseudosamples).

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth target values.
            If hierarchical: row ``MultiIndex`` where last level is the time index
            and preceding levels are hierarchy levels.
            If non-hierarchical: pd.DataFrame with RangeIndex or DatetimeIndex.

        y_pred : pd.DataFrame
            Predicted values to evaluate against ground truth.
            Same format and indices as ``y_true``.

        **kwargs : dict, optional
            Optional keyword arguments, may include:

            y_train : pd.DataFrame
                Training data, required if ``requires-y-train`` tag is True.
            y_pred_benchmark : pd.DataFrame, same format as y_pred
                Benchmark predictions, required if tag
                ``requires-y-pred-benchmark`` is True.
            sample_weight : 1D array-like, default=None
                Per-time-point sample weights.

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric result. Return type depends on ``multioutput``
            and ``multilevel``:

            * float
              if ``multioutput="uniform_average"`` or array-like,
              and ``multilevel`` in ``["uniform_average", "uniform_average_time"]``.
              Value is the metric averaged over variables and levels.

            * ``np.ndarray`` of shape ``(len(y_true.columns),)``
              if ``multioutput="raw_values"``
              and ``multilevel`` in ``["uniform_average", "uniform_average_time"]``.
              i-th entry is the metric for the i-th output variable.

            * ``pd.DataFrame`` if ``multilevel="raw_values"``.
              Shape ``(n_levels, 1)`` if ``multioutput="uniform_average"``.
              Shape ``(n_levels, len(y_true.columns))`` if ``multioutput="raw_values"``.
              Metric is computed per hierarchy level; columns averaged per multioutput.
        """
        raise NotImplementedError("abstract method")
        # todo: implement your hierarchical metric logic here.
        # The example below shows the structure for handling hierarchical data.
        # Replace with your own logic.
        #
        # multioutput = self.multioutput
        # multilevel = self.multilevel
        #
        # --- case: multilevel leads to averaging (not "raw_values") ---
        # if multilevel != "raw_values":
        #     raw_values = (y_true - y_pred).abs()
        #     raw_values = self._get_weighted_df(raw_values, **kwargs)
        #     result = raw_values.mean(axis=0)
        #     return self._handle_multioutput(result, multioutput)
        #
        # --- case: multilevel="raw_values", return per-level results ---
        # else:
        #     level_names = y_true.index.names[:-1]
        #     grouped = y_true.groupby(level=level_names)
        #     results = {}
        #     for group_key, group_idx in grouped.groups.items():
        #         y_true_g = y_true.loc[group_idx]
        #         y_pred_g = y_pred.loc[group_idx]
        #         raw_values_g = (y_true_g - y_pred_g).abs()
        #         raw_values_g = self._get_weighted_df(raw_values_g, **kwargs)
        #         results[group_key] = raw_values_g.mean(axis=0)
        #     out_df = pd.DataFrame(results).T
        #     out_df.index.names = level_names
        #     if isinstance(multioutput, str) and multioutput == "raw_values":
        #         return out_df
        #     elif isinstance(multioutput, str) and multioutput == "uniform_average":
        #         return out_df.mean(axis=1).to_frame()
        #     else:
        #         return out_df.dot(multioutput).to_frame()

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Optional to implement. If not implemented, defaults to jackknife
        pseudosamples derived from ``_evaluate``.

        Override for efficiency or if per-time-point evaluation has a
        closed-form expression for your metric.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth (may have MultiIndex for hierarchical data).
        y_pred : pd.DataFrame
            Predicted values, same format as y_true.
        **kwargs : dict, optional
            Same optional kwargs as ``_evaluate``.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Metric at each time point.
            pd.Series if multioutput="uniform_average" or array-like.
            pd.DataFrame if multioutput="raw_values".
            Index matches y_true.index (including hierarchy levels if present).
        """
        # default: jackknife pseudosamples from _evaluate
        # override for closed-form per-index computation
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
        # todo: replace with your actual test parameters
        params1 = {"parama": 1, "paramb": "default"}
        params2 = {"parama": 2, "paramb": "option2", "multilevel": "raw_values"}
        return [params1, params2]
