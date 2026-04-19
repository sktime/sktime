"""Extension for template of hierarchical forecasting metrics.

Purpose of this implementation template:
    Quick implementation of new estimators following the template
    This is not a concrete class or Base class to import!
    Use this as a starting template to build on.

    This template is for metrics that need to handle hierarchical data
    internally, for example to perform cross-level aggregations or computations
    that depend on the hierarchy structure.

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
        evaluating            - _evaluate(self, y_true, y_pred, **kwargs)

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

import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

# todo: add any necessary imports here


# todo: change class name and write docstring
class MyForecastingHierarchicalMetric(BaseForecastingErrorMetric):
    """Custom hierarchical forecasting metric. todo: write docstring.

    todo: describe your custom metric here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    multioutput : {'uniform_average', 'raw_values'} or array-like, default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
    multilevel : {'uniform_average_time', 'uniform_average', 'raw_values'},
        default='uniform_average_time'
        Defines how to aggregate metric for hierarchical data (with levels).
        In this template, we default to 'uniform_average_time' to ensure the
        Base class passes the full MultiIndex DataFrame to _evaluate.
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
        # signals that this metric handles hierarchy internally
        "inner_implements_multilevel": True,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        parama,
        multioutput="uniform_average",
        multilevel="uniform_average_time",
        by_index=False,
    ):
        self.parama = parama

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate.

        When "inner_implements_multilevel" is True and multilevel="uniform_average_time",
        this method receives the full MultiIndex DataFrame (hierarchical data).

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth (correct) target values.
            Can have MultiIndex for hierarchical data.
        y_pred : pd.DataFrame
            Predicted values to evaluate.
            Must match format of y_true.
        kwargs : dict, optional
            Additional keyword arguments passed to the metric function.

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric.
            Aggregation is controlled by self.multioutput and self.multilevel.
        """
        # todo: implement your metric logic here

        # Example of handling hierarchy:
        # if isinstance(y_true.index, pd.MultiIndex):
        #     # Extract hierarchy levels (excluding time level at the end)
        #     # hierarchy_levels = list(range(y_true.index.nlevels - 1))
        #
        #     # compute some metric per hierarchy level using groupby
        #     # errors = (y_true - y_pred).abs()
        #     # per_level_loss = errors.groupby(level=hierarchy_levels).mean()
        #
        #     # Handle self.multilevel aggregation:
        #     # if self.multilevel == "raw_values":
        #     #     return self._handle_multioutput(per_level_loss, self.multioutput)
        #     # elif self.multilevel in ["uniform_average", "uniform_average_time"]:
        #     #     final_loss = per_level_loss.mean()
        #     #     return self._handle_multioutput(final_loss, self.multioutput)

        # If no MultiIndex, fall back to flat computation or error

        raise NotImplementedError("Abstract method.")

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
        params2 = {"parama": 2, "multilevel": "raw_values"}
        return [params1, params2]
