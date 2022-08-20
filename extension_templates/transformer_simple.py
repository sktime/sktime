# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Extension template for transformers, SIMPLE version.

Contains only bare minimum of implementation requirements for a functional transformer.
Also assumes *no composition*, i.e., no transformer or other estimator components.
For advanced cases (inverse transform, composition, etc),
    see full extension template in forecasting.py

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
- more details: https://www.sktime.org/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, X, y=None)
    transformation  - _transform(self, X, y=None)

Testing - implement if sktime transformer (not needed locally):
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
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        #   not relevant for transformers that return Primitives in transform-output
        "handles-missing-data": False,  # can estimator handle missing data?
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):

        # todo: write any hyper-parameters to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

        # todo: change "MyTransformer" to the name of the class
        super(MyTransformer, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

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
        #
        # special case: if no fitting happens before transformation
        #  then: delete _fit (don't implement)
        #   set "fit_is_empty" tag to True
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
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
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
