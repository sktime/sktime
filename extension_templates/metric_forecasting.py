"""Extension for template of forecasting metrics.

Purpose of this implementation template:
    Quick implementation of new estimators following the template
    This is not a concrete class or Base class to import!
    Use this as a starting template to build on.

    How to use:
    - Copy the template in the suitable folder and give a descriptive name
    - Work through all the todo comments given
    - Ensure to implement the mandatory methods
    - Do not write in reserved variables: _tags
    - you can add more private methods, but do not override BaseEstimator's
    private methods an easy way to be safe is to prefix your methods with "_custom"
    - change docstrings for functions and the file
    - ensure interface compatibility by testing performance_metrics/forecasting/tests
    - once complete: use as a local library, or contribute to sktime via PR
    - more details:
      https://www.sktime.net/en/stable/developer_guide/add_estimators.html

    Mandatory methods to implement:
        evaluating            - _evaluate_by_index(self, y_true, y_pred, **kwargs)

    Testing - required for sktime test framework and check_estimator usage:
        get default parameters for test instance(s) - get_test_params()

    copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class MyForecastingMetric(BaseForecastingErrorMetric):
    """Custom forecasting metric. todo: write docstring.

    todo: describe your custom metric here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    multioutput : {'uniform_average', 'raw_values'} or array-like, default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        * If 'uniform_average' (default), errors of all outputs are averaged
          with uniform weight.
        * If 'raw_values', per-variable errors are returned.
        * If array-like, errors are averaged with values used as weights.
    multilevel : {'uniform_average', 'uniform_average_time', 'raw_values'},
        default='uniform_average'
        Defines how to aggregate metric for hierarchical data (with levels).
        * If 'uniform_average' (default), errors are mean-averaged across levels.
        * If 'uniform_average_time', metric is applied to all data, ignoring levels.
        * If 'raw_values', does not average across levels, hierarchy is retained.
    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.
    """

    # optional todo: override base class estimator default tags here if necessary
    # these are the default values, only add if different to these.
    _tags = {
        # tags and full specifications are available in the tag API reference
        # https://www.sktime.net/en/stable/api_reference/tags.html
        #
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        # specify one or multiple authors and maintainers, only for sktime contribution
        # remove maintainer tag if maintained by sktime core team
        # estimator tags
        # --------------
        "object_type": ["metric_forecasting", "metric"],
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "capability:multivariate": True,
        "lower_is_better": True,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        parama,
        paramb="default",
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        self.parama = parama
        self.paramb = paramb

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth (correct) target values.
        y_pred : pd.DataFrame
            Predicted values to evaluate.
        kwargs : dict, optional
            Additional keyword arguments passed to the metric function.
            Can include:
            - y_pred_benchmark : pd.DataFrame, same format as y_pred
              Required if tag "requires-y-pred-benchmark" is True.
            - y_train : pd.DataFrame, same format as y_true
              Required if tag "requires-y-train" is True.
            - sample_weight : 1D array-like, or callable, default=None

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.
            pd.Series if self.multioutput="uniform_average" or array-like.
            pd.DataFrame if self.multioutput="raw_values".
        """
        # todo: implement your metric logic here
        # Example using MAE-like logic:
        # absolute_errors = (y_true - y_pred).abs()

        # You should use helper methods to handle sample_weight and multioutput:
        # weighted_df = self._get_weighted_df(absolute_errors, **kwargs)
        # result = self._handle_multioutput(weighted_df, self.multioutput)

        # return result
        raise NotImplementedError("Abstract method.")

    # optional: override _evaluate if the metric has special aggregation needs
    # default implementation is result = self._evaluate_by_index(y_true, y_pred, **kwargs).mean()
    # def _evaluate(self, y_true, y_pred, **kwargs):
    #     ...

    # todo: return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        # todo: set the testing parameters for the estimators
        params1 = {"parama": 1}
        params2 = {"parama": 2, "paramb": "other"}
        return [params1, params2]
