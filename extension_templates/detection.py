"""Extension template for series detection - outliers, changepoints, segments.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
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

Mandatory implements:
    fitting         - _fit(self, X, y=None)
    annotating     - _predict(self, X)

Optional implements:
    updating        - _update(self, X, y=None)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.detection.base import BaseDetector

# todo: add any necessary imports here


class MyDetector(BaseDetector):
    """Custom time series detector for anomalies, change points, or segments.

    todo: write docstring, describing your custom forecaster

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

    _tags = {
        # tags and full specifications are available in the tag API reference
        # https://www.sktime.net/en/stable/api_reference/tags.html
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # estimator tags
        # --------------
        #
        # detection tasks fall into categories including anomaly or outlier detection,
        # change point detection, and time series segmentation and segment detection
        "task": "segmentation",
        # valid values: "change_point_detection", "anomaly_detection", "segmentation"
        #
        # learning_type = learning type of the detection task
        "learning_type": "unsupervised",
        # valid values: "unsupervised", "supervised", "semi_supervised"
        #
        # capability:multivariate controls whether internal X can be multivariate
        # if True (only univariate), always applies vectorization over variables
        "capability:multivariate": False,
        # valid values: True = inner _fit, _transform receive only univariate series
        #   False = uni- and multivariate series are passed to inner methods
        #
        # fit_is_empty = is fit empty and can be skipped?
        "fit_is_empty": True,
        # valid values: True = _fit is considered empty and skipped, False = No
        # CAUTION: default is "True", i.e., _fit will be skipped even if implemented
        #
        # capability:missing_data = can estimator handle missing data?
        "capability:missing_data": False,
        # valid values: boolean True (yes), False (no)
        # if False, raises exception if y or X passed contain missing data (nans)
        #
        # X_inner_mtype control which format X appears in in the inner functions _fit,
        # _predict, etc
        "X_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        "distribution_type": "None",  # Tag to determine test in test_all_annotators
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of the estimator
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
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
    def __init__(self, parama, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

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
    def _fit(self, X, y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.

        y : pd.DataFrame with RangeIndex
            Known events for training, in ``X``, if detector is supervised.

            Each row ``y`` is a known event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.

        Returns
        -------
        self :
            Reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: implement this, mandatory
    def _predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with RangeIndex
            Detected or predicted events.

            Each row ``y`` is a detected or predicted event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    def _update(self, X, y=None):
        """Update model with new data and optional ground truth labels.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to update model with, time series
        y : pd.Series, optional
            ground truth detection labels for training, if detector is supervised

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        updates fitted model (attributes ending in "_")
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh

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
            There are currently no reserved values for annotators.

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
