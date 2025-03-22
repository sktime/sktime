"""Extension for template of detection metrics.

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
    - ensure interface compatibility by testing performance_metrics/detection/tests
    - once complete: use as a local library, or contribute to sktime via PR
    - more details:
      https://www.sktime.net/en/stable/developer_guide/add_estimators.html

    Mandatory implements:
        evaluating            - _evaluate(self, X)

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

from sktime.performance_metrics.detection._base import BaseDetectionMetric

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class MyMetric(BaseDetectionMetric):
    """Custom metric. todo: write docstring.

    todo: describe your custom metric here

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
        #     if interfacing a 3rd party estimator, ensure to give credit to the
        #     authors of the interfaced estimator
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        #     for 3rd party interfaces, the scope is the sktime class only
        # specify one or multiple authors and maintainers, only for sktime contribution
        # remove maintainer tag if maintained by sktime core team
        # estimator tags
        # --------------
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "requires_y_true": True,  # if False, is unsupervised metric
        "lower_is_better": True,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama

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

    def _evaluate(self, y_true, y_pred, X):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate.

        Parameters
        ----------
        y_true :pd.DataFrame
                time series in ``sktime`` compatible data container format.
                Ground truth (correct) event locations, in ``X``\
                Should only be ``pd.DataFrame``.
                Expected format:
                    Index: time indices or event identifiers
                    Columns: depending on scitype (`points` or `segments`).
                    `points` assumes single column, `segments` require ["start","end"].
            For further details on data format, see glossary on :term:`mtype`.

        y_pred :pd.DataFrame
                time series in ``sktime`` compatible data container format \
                Detected events to evaluate against ground truth. \
                Must be same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that \
            values in ``X`` do not matter.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        raise NotImplementedError("Abstract method.")
        # implement here
        # IMPORTANT: avoid side effects to y_pred and y_true
        # You can change the definition of y_pred and y_true based on requirement,
        # but always mention it in the docstring

    # todo: return default parameters, so that a test instance can be created
    # required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for clusterers.

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
