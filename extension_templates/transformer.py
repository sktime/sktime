# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for transformers.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y,
    _converter_store_X, transformers_, _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, X, y=None)
    transformation  - _transform(self, X, y=None)

Optional implements:
    inverse transformation      - _inverse_transform(self, X, y=None)
    update                      - _update(self, X, y=None)
    fitted parameter inspection - _get_fitted_params()

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

# todo: add any necessary sktime external imports here

from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class MyTransformer(BaseTransformer):
    """Custom transformer. todo: write docstring.

    todo: describe your custom transformer here
        fill in sections appropriately
        docstring must be numpydoc compliant

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    est : sktime.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    #
    # todo: define the transformer scitype by setting the tags
    #   scitype:transform-input - the expected input scitype of X
    #   scitype:transform-output - the output scitype that transform produces
    #   scitype:transform-labels - whether y is used and if yes which scitype
    #   scitype:instancewise - whether transform uses all samples or acts by instance
    #
    # todo: define internal types for X, y in _fit/_transform by setting the tags
    #   X_inner_mtype - the internal mtype used for X in _fit and _transform
    #   y_inner_mtype - if y is used, the internal mtype used for y; usually "None"
    #   setting this guarantees that X, y passed to _fit, _transform are of above types
    #   for possible mtypes see datatypes.MTYPE_REGISTER, or the datatypes tutorial
    #
    #  when scitype:transform-input is set to Panel:
    #   X_inner_mtype must be changed to one or a list of sktime Panel mtypes
    #  when scitype:transform-labels is set to Series or Panel:
    #   y_inner_mtype must be changed to one or a list of compatible sktime mtypes
    #  the other tags are "safe defaults" which can usually be left as-is
    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="transformer", as_dataframe=True)
        #
        #
        # behavioural tags: transformer type
        # ----------------------------------
        #
        # scitype:transform-input, scitype:transform-output, scitype:transform-labels
        # control the input/output type of transform, in terms of scitype
        #
        # scitype:transform-input, scitype:transform-output should be the
        # simplest scitype that describes the mapping, taking into account vectorization
        # a transform that produces Series when given Series, Panel when given Panel
        #   should have both transform-input and transform-output as "Series"
        # a transform that produces a tabular DataFrame (Table)
        #   when given Series or Panel should have transform-input "Series"
        #       and transform-output as "Primitives"
        "scitype:transform-input": "Series",
        # valid values: "Series", "Panel"
        "scitype:transform-output": "Series",
        # valid values: "Series", "Panel", "Primitives"
        #
        # scitype:instancewise = is fit_transform an instance-wise operation?
        # instance-wise = only values of a given series instance are used to transform
        #   that instance. Example: Fourier transform; non-example: series PCA
        "scitype:instancewise": True,
        #
        # scitype:transform-labels types the y used in transform
        #   if y is not used in transform, this should be "None"
        "scitype:transform-labels": "None",
        # valid values: "None" (not needed), "Primitives", "Series", "Panel"
        #
        #
        # behavioural tags: internal type
        # ----------------------------------
        #
        # X_inner_mtype, y_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _transform, etc
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   y_inner_mtype can also be of scitype Table (one row/instance per series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # univariate-only controls whether internal X can be univariate/multivariate
        # if True (only univariate), always applies vectorization over variables
        "univariate-only": False,
        # valid values: True = inner _fit, _transform receive only univariate series
        #   False = uni- and multivariate series are passed to inner methods
        #
        # requires_X = does X need to be passed in fit?
        "requires_X": True,
        # valid values: False (no), True = exception is raised if no X is seen in _fit
        #   requires_y setting is independent of requires_X
        #
        # requires_y = does y need to be passed in fit?
        "requires_y": False,
        # valid values: False (no), True = exception is raised if no y is seen in _fit
        #   requires_X setting is independent of requires_y
        #
        # remember_data = whether all data seen is remembered as self._X
        "remember_data": False,
        # valid vales: False (no), True = self._X is created/update in fit/update
        #   self._X is all X passed via fit or update, updated via update_data
        #   self._X is of mtype seen in fit, update adds more data to the same container
        #   self._X can be used (readonly) by the estimator in _fit, _transform, _update
        #   if set to True, fit-is-empty must be set to False
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # fit_is_empty = is fit empty and can be skipped?
        "fit_is_empty": True,
        # valid values: True = _fit is considered empty and skipped, False = No
        # CAUTION: default is "True", i.e., _fit will be skipped even if implemented
        #
        # X-y-must-have-same-index = can estimator handle different X/y index?
        "X-y-must-have-same-index": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception if X.index is not contained in y.index
        #
        # enforce_index_type = index type that needs to be enforced in X/y
        "enforce_index_type": None,
        # valid values: pd.Index subtype, or list of pd.Index subtype
        # if not None, raises exception if X.index, y.index level -1 is not of that type
        #
        # transform-returns-same-time-index = does transform return same index as input?
        "transform-returns-same-time-index": False,
        # valid values: boolean True (yes), False (no)
        # if True, transform and inverse_transform returns should have
        #   same length and same index (if pandas) as inputs
        # no exception is raised if this tag is incorrectly set
        #
        # capability:inverse_transform = is inverse_transform implemented?
        "capability:inverse_transform": False,
        # valid values: boolean True (yes), False (no)
        # if True, _inverse_transform must be implemented
        # if False, exception is raised if inverse_transform is called,
        #   unless the skip-inverse-transform tag is set to True
        #
        # capability:inverse_transform:range = domain of invertibility of transform
        "capability:inverse_transform:range": None,
        # valid values: None (no range), list of two floats [min, max]
        # if None, inverse_transform is assumed to be defined for all values
        # if list of floats, invertibility is assumed
        # only in the closed interval [min, max] of transform
        # note: the range applies to the *input* of transform, not the output
        #
        # capability:inverse_transform:exact = is inverse transform exact?
        "capability:inverse_transform:exact": True,
        # valid values: boolean True (yes), False (no)
        # if True, inverse_transform is assumed to be exact inverse of transform
        # if False, inverse_transform is assumed to be an approximation
        #
        # skip-inverse-transform = is inverse-transform skipped when called?
        "skip-inverse-transform": False,
        # if False, capability:inverse_transform tag behaviour is as per default
        # if True, inverse_transform is the identity transform and raises no exception
        #   this is useful for transformers where inverse_transform
        #   may be called but should behave as the identity, e.g., imputers
        #
        # capability:unequal_length = can the transformer handle unequal length panels,
        #   i.e., when passed unequal length instances in Panel or Hierarchical data
        "capability:unequal_length": True,
        # valid values: boolean True (yes), False (no)
        # if False, may raise exception when passed unequal length Panel/Hierarchical
        #
        # capability:unequal_length:removes = if passed Panel/Hierarchical,
        #   is transform result always guaranteed to be equal length (and series)?
        "capability:unequal_length:removes": False,
        # valid values: boolean True (yes), False (no)
        # applicable only if scitype:transform-output is not "Primitives"
        # used for search index and validity checking, does not raise direct exception
        #
        # handles-missing-data = can the transformer handle missing data (np or pd.NA)?
        "handles-missing-data": False,  # can estimator handle missing data?
        # valid values: boolean True (yes), False (no)
        # if False, may raise exception when passed time series with missing values
        #
        # capability:missing_values:removes = if passed time series
        #   is transform result always guaranteed to contain no missing values?
        "capability:missing_values:removes": False,
        # valid values: boolean True (yes), False (no)
        # used for search index and validity checking, does not raise direct exception
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of th estimator
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
    # in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__
    #  avoid if possible, but see __init__ for instructions when needed

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, est, parama, est2=None, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.est = est
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama
        # for estimators, initialize a clone, e.g., self.est_ = est.clone()

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.est2 = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory (except in special case below)
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """

        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals, should be used or fitted if needed
        #
        # special case: if no fitting happens before transformation
        #  then: delete _fit (don't implement)
        #   set "fit_is_empty" tag to True
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # if transform-output is "Primitives":
        #  return should be pd.DataFrame, with as many rows as instances in input
        #  if input is a single series, return should be single-row pd.DataFrame
        # if transform-output is "Series":
        #  return should be of same mtype as input, X_inner_mtype
        #  if multiple X_inner_mtype are supported, ensure same input/output
        # if transform-output is "Panel":
        #  return a multi-indexed pd.DataFrame of Panel mtype pd_multiindex
        #
        # todo: add the return mtype/scitype to the docstring, e.g.,
        #  Returns
        #  -------
        #  X_transformed : Series of mtype pd.DataFrame
        #       transformed version of X

    # todo: consider implementing this, optional
    # if not implementing, delete the _inverse_transform method
    # inverse transform exists only if transform does not change scitype
    #  i.e., Series transformed to Series
    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        # implement here
        # IMPORTANT: avoid side effects to X, y
        #
        # type conventions are exactly those in _transform, reversed
        #
        # for example: if transform-output is "Series":
        #  return should be of same mtype as input, X_inner_mtype
        #  if multiple X_inner_mtype are supported, ensure same input/output
        #
        # todo: add the return mtype/scitype to the docstring, e.g.,
        #  Returns
        #  -------
        #  X_inv_transformed : Series of mtype pd.DataFrame
        #       inverse transformed version of X

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    # standard behaviour is "no update"
    # also delete in the case where there is no fitting
    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _update must support all types in it
            Data to update transformer with
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: reference to self
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals, should be used or fitted if needed

    # todo: consider implementing this, optional
    # implement only if different from default:
    #   default retrieves all self attributes ending in "_"
    #   and returns them with keys that have the "_" removed
    # if not implementing, delete the method
    #   avoid overriding get_fitted_params
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

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
            There are currently no reserved values for transformers.

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
