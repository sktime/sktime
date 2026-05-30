"""Extension template for forecasting performance metrics - hierarchical.

Purpose of this implementation template:
    Quick implementation of new hierarchical forecasting metric estimators.
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

Difference from metric_forecasting.py (non-hierarchical template):
    This template sets the tag "inner_implements_multilevel" to True.
    The metric receives the full pd.DataFrame with MultiIndex (including hierarchy
    levels) and is responsible for handling aggregation across hierarchy levels
    according to self.multilevel ("uniform_average", "uniform_average_time",
    "raw_values"). Use this when the metric cannot be computed independently per
    hierarchy node, e.g., reconciliation metrics that compare hierarchy levels.

    For metrics that CAN be computed independently per hierarchy node,
    use the non-hierarchical template (metric_forecasting.py) instead, which
    uses the base class to automatically vectorize across hierarchy levels.

How to use this implementation template to implement a new metric:
- Make a copy of the template in a suitable location, give it a descriptive name.
- Work through all the "todo" comments below.
- Fill in code for mandatory methods; see "Mandatory methods" below.
- You can add more private methods, but do not override private methods of the base.
- Change docstrings for functions and the file.
- Ensure interface compatibility by sktime.utils.estimator_checks.check_estimator.
- Once complete: use as a local library, or contribute to sktime via PR.
- More details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory methods to implement (at least one of):
    overall evaluation         - _evaluate(self, y_true, y_pred, **kwargs)
    per-time-index evaluation  - _evaluate_by_index(self, y_true, y_pred, **kwargs)

    For hierarchical metrics, prefer implementing _evaluate natively.
    The base class default _evaluate_by_index uses jackknife pseudosamples from
    _evaluate.

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: add an appropriate copyright notice for your estimator

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

# todo: add any necessary imports here


# todo: change class name and write docstring
class MyForecastingMetricHierarchical(BaseForecastingErrorMetric):
    """Custom hierarchical forecasting metric. todo: write docstring.

    todo: describe your custom hierarchical metric here.

    Use this template when the metric must access data across hierarchy levels
    simultaneously, e.g., reconciliation errors or cross-level aggregation metrics.
    For metrics computed independently per hierarchy node, use metric_forecasting.py.

    Parameters named multioutput, multilevel, by_index are inherited from the base
    class and should not be redeclared; see their descriptions in
    BaseForecastingErrorMetric.

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : str, optional (default='default')
        descriptive explanation of paramb

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from extension_templates.metric_forecasting_hierarchical import (
    ...     MyForecastingMetricHierarchical,
    ... )
    >>> idx = pd.MultiIndex.from_product(
    ...     [["a", "b"], ["x"], range(5)], names=["h1", "h2", "time"]
    ... )
    >>> y_true = pd.DataFrame({"y": np.arange(10.0)}, index=idx)
    >>> y_pred = pd.DataFrame({"y": np.arange(10.0) + 0.1}, index=idx)
    >>> metric = MyForecastingMetricHierarchical(parama=1)
    >>> metric(y_true, y_pred)
    """

    # optional todo: override base class estimator default tags here if necessary
    # these are the default values; only add tags that differ from the base defaults.
    # tag descriptions are at: https://www.sktime.net/en/stable/api_reference/tags.html
    _tags = {
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        # remove maintainer tag if maintained by sktime core team
        #
        # estimator tags
        # --------------
        "object_type": ["metric_forecasting", "metric"],
        # Setting inner_implements_multilevel = True tells the base class that this
        # metric handles the full hierarchical MultiIndex data internally.
        # _evaluate / _evaluate_by_index will receive pd.DataFrame with MultiIndex.
        "inner_implements_multilevel": True,
        # "requires-y-train": False,
        # "requires-y-pred-benchmark": False,
        # "capability:multivariate": True,
        # "lower_is_better": True,
    }

    # todo: add any hyper-parameters to constructor; do NOT redeclare
    #       multioutput, multilevel, by_index - these are inherited
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
        # IMPORTANT: do not overwrite or mutate self.parama etc. from now on

        # leave this as is
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

        # do not put anything else in __init__,
        # use __post_init__ for any further initialization logic

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes."""
        # todo: optional, parameter checking or coercion logic here

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Return the metric value, averaged over time.

        private _evaluate containing core logic, called from evaluate.

        For hierarchical metrics (inner_implements_multilevel=True) this method
        receives the full MultiIndex DataFrame and must handle multilevel aggregation
        according to self.multilevel.

        Mandatory to implement if _evaluate_by_index is not implemented.

        Parameters
        ----------
        y_true : pd.DataFrame with row MultiIndex (hierarchy levels + time as last)
            Ground truth (correct) target values.
            Row index is a pd.MultiIndex; the last level is the time index.
            Columns are the output variables.

        y_pred : pd.DataFrame with row MultiIndex
            Predicted values to evaluate. Same index/columns as y_true.

        **kwargs
            sample_weight : 1D array-like, optional
                Per-time-point weights (same length as time axis per instance).

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            float         if multioutput != "raw_values" and multilevel != "raw_values"
            np.ndarray    if multioutput == "raw_values" and multilevel != "raw_values"
                          shape (n_variables,)
            pd.DataFrame  if multilevel == "raw_values"
                          shape (n_levels, 1)       if multioutput != "raw_values"
                          shape (n_levels, n_vars)  if multioutput == "raw_values"
        """
        # todo: implement hierarchical metric evaluation here.
        # Do NOT call super().__call__ or evaluate - that causes infinite recursion.
        #
        # y_true and y_pred arrive as pd.DataFrame with row MultiIndex.
        # The last index level is time; preceding levels are hierarchy levels.
        #
        # You must handle self.multilevel:
        #   "uniform_average"      -> return float or np.ndarray (averaged over levels)
        #   "uniform_average_time" -> metric applied ignoring hierarchy, same return
        #   "raw_values"           -> return pd.DataFrame, one row per hierarchy group
        #
        # You must handle self.multioutput:
        #   "uniform_average" / array-like -> aggregate across columns
        #   "raw_values"                   -> return per-column values
        #
        # Typical pattern outline:
        #
        #   multilevel = self.multilevel
        #   multioutput = self.multioutput
        #
        #   if multilevel == "uniform_average_time":
        #       # treat all rows as a flat time series, ignore level structure
        #       raw_values = (y_true - y_pred).abs()
        #       raw_values = self._get_weighted_df(raw_values, **kwargs)
        #       return self._handle_multioutput(raw_values.mean(axis=0), multioutput)
        #
        #   # group by all hierarchy levels except time (last level)
        #   level_names = y_true.index.names[:-1]
        #   grouped = (y_true - y_pred).abs().groupby(level=level_names)
        #   per_level = grouped.mean()   # pd.DataFrame indexed by hierarchy groups
        #
        #   if multilevel == "raw_values":
        #       return self._handle_multioutput(per_level, multioutput)
        #   else:  # "uniform_average"
        #       averaged = per_level.mean(axis=0)
        #       return self._handle_multioutput(averaged.to_frame().T, multioutput)
        #
        # IMPORTANT: avoid side effects to y_true, y_pred

        raise NotImplementedError("abstract method")

    # todo: alternatively or additionally implement _evaluate_by_index:
    #
    # def _evaluate_by_index(self, y_true, y_pred, **kwargs):
    #     """Return the metric evaluated at each time point.
    #
    #     Same as _evaluate but returns per-index values instead of an overall scalar.
    #     y_true / y_pred have the same MultiIndex format as in _evaluate.
    #     Return pd.Series (uniform_average) or pd.DataFrame (raw_values)
    #     with the same index as y_true.
    #     """
    #     raise NotImplementedError("abstract method")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # todo: set the testing parameters for the estimator
        # All required constructor parameters must be specified.

        params1 = {"parama": 1}
        params2 = {"parama": 2, "paramb": "other"}

        return [params1, params2]
