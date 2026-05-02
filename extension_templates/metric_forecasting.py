"""Extension template for forecasting performance metrics - non-hierarchical.

Purpose of this implementation template:
    Quick implementation of new forecasting metric estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

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
    per-time-index evaluation  - _evaluate_by_index(self, y_true, y_pred, **kwargs)
    overall evaluation         - _evaluate(self, y_true, y_pred, **kwargs)

    The base class default _evaluate calls _evaluate_by_index and averages.
    The base class default _evaluate_by_index uses jackknife pseudosamples from
    _evaluate.
    Implement at least one; implementing both is optimal.

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
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

    For non-hierarchical metrics that can be computed independently per
    hierarchy node. For metrics that require full hierarchy access, use
    metric_forecasting_hierarchical.py.

    Parameters named multioutput, multilevel, by_index are inherited from the base
    class and should not be redeclared in the signature; see their descriptions in
    BaseForecastingErrorMetric.

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : str, optional (default='default')
        descriptive explanation of paramb

    Examples
    --------
    >>> import pandas as pd
    >>> from extension_templates.metric_forecasting import MyForecastingMetric
    >>> y_true = pd.Series([3, -0.5, 2, 7, 2])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8, 1.25])
    >>> metric = MyForecastingMetric(parama=1)
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
        # "scitype:y_pred": "pred",  # pred/pred_interval/pred_quantiles/pred_var
        # "requires-y-train": False,   # does metric require in-sample y_train?
        # "requires-y-pred-benchmark": False,  # requires benchmark y_pred?
        # "capability:multivariate": True,  # can handle multivariate y?
        # "lower_is_better": True,  # lower value = better metric?
        # "inner_implements_multilevel": False,  # base handles multilevel via vectorize
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
        # write derived/default values to self._parama, self._newparam (with _)

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
        # example of handling a mutable default:
        # if self.parama is None:
        #     self._parama = SomeDefaultObject()
        # else:
        #     self._parama = self.parama.clone()

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index.

        Mandatory to implement if _evaluate is not implemented.
        If implemented, the base class _evaluate defaults to the mean of this.

        Parameters
        ----------
        y_true : pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime pd.DataFrame format for Series scitype.

        y_pred : pd.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate.
            Time series in sktime pd.DataFrame format for Series scitype.

        **kwargs
            sample_weight : 1D array-like, optional
                Per-time-point sample weights. Use self._get_weighted_df(df, **kwargs)
                to apply weights to a raw-value DataFrame.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.

            pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        # todo: implement per-index metric logic here.
        # Do NOT call super().__call__ or evaluate - that causes infinite recursion.
        #
        # y_true and y_pred arrive as pd.DataFrame (already coerced by base class).
        #
        # Typical pattern (example: absolute error):
        #
        #   multioutput = self.multioutput
        #   raw_values = (y_true - y_pred).abs()         # pd.DataFrame, same shape
        #   raw_values = self._get_weighted_df(raw_values, **kwargs)  # apply weights
        #   return self._handle_multioutput(raw_values, multioutput)
        #
        # _handle_multioutput returns:
        #   pd.Series  if multioutput in ("uniform_average", array-like)
        #   pd.DataFrame if multioutput == "raw_values"
        #
        # IMPORTANT: avoid side effects to y_true, y_pred

        raise NotImplementedError("abstract method")

    # todo: alternatively or additionally implement _evaluate for efficiency:
    #
    # def _evaluate(self, y_true, y_pred, **kwargs):
    #     """Return overall metric value, averaged over time.
    #
    #     private _evaluate containing core logic, called from evaluate.
    #     The base class default calls _evaluate_by_index and averages.
    #     Only override this if computing the average directly is more efficient.
    #
    #     Parameters
    #     ----------
    #     y_true : pd.DataFrame   (same format as _evaluate_by_index)
    #     y_pred : pd.DataFrame   (same format as _evaluate_by_index)
    #
    #     Returns
    #     -------
    #     loss : float or np.ndarray
    #         float if multioutput="uniform_average" or array-like
    #         np.ndarray of shape (n_variables,) if multioutput="raw_values"
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
        # Testing parameters can be a dictionary or list of dictionaries.
        # All required constructor parameters must be specified.
        # At least two parameter sets are recommended for coverage.
        #
        # example 1: single dictionary
        # params = {"parama": 1, "paramb": "default"}
        #
        # example 2: list of dictionaries
        # params = [{"parama": 1}, {"parama": 2, "paramb": "other"}]

        params1 = {"parama": 1}
        params2 = {"parama": 2, "paramb": "other"}

        return [params1, params2]
