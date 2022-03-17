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
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by testing transformations/tests/test_all_transformers
        and tests/test_all_estimators
- once complete: use as a local library, or contribute to sktime via PR

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


__author__ = ["ciaran-g", "eenticott-shell", "k1m190r"]

import numpy as np
import pandas as pd
from numpy.linalg import inv

from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here


class reconciler(BaseTransformer):
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
        "scitype:transform-input": "Hierarchical",
        "scitype:transform-output": "Hierarchical",
        "scitype:transform-labels": "None",
        # todo instance wise?
        "scitype:instancewise": True,  # is this an instance-wise transform?
        # which mtypes do _fit/_predict support for X?
        "X_inner_mtype": "pd_multiindex_hier",
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": True,  # index type that needs to be enforced in X/y
        "fit-in-transform": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
    }

    # test that method is recognised
    def __init__(self, method="bu"):
        self.method = method
        # self._g_dispatch = {
        #     "bu": _get_g_matrix_bu,
        #     "ols": _get_g_matrix_ols,
        #     # "wls_str": self.get_g_matrix_wls_str,
        # }
        super(reconciler, self).__init__()

    # todo: implement this, mandatory (except in special case below)
    # test for type of input?
    # test for __total present in index?
    # tests for index matching at each time point?
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
        #   set "fit-in-transform" tag to True

        # self.hier_data = X

        # TODO add checks for if method exists in our dict
        # get g matrix function
        # method_fn = self._g_dispatch[self.method]

        # define matrices
        # self.g_matrix = self._g_dispatch[self.method](X)
        if self.method == "bu":
            self.g_matrix = _get_g_matrix_bu(X)
        elif self.method == "ols":
            self.g_matrix = _get_g_matrix_ols(X)
        else:
            self.g_matrix = _get_g_matrix_wls_str(X)

        self.s_matrix = _get_s_matrix(X)

        return self

    # todo: implement this, mandatory
    # tests for index matching?
    # tests for actually hierarchical predictions?
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
        X = X.groupby(level="timepoints")
        recon_preds = X.transform(lambda x: _reconcile(x, self.s_matrix, self.g_matrix))

        return recon_preds

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

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
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
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


# include index between matrices here as in df.dot()
def _reconcile(base_fc, s_matrix, g_matrix):
    # return s_matrix.dot(g_matrix.dot(base_fc))
    return np.dot(s_matrix, np.dot(g_matrix, base_fc))


# tests for matrix index?
# what happens if two end-points have the same name?
def _get_s_matrix(X):
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel("timepoints")
        .unique()
    )
    # get all level indexes
    al_inds = X.droplevel(level="timepoints").index.unique()

    s_matrix = pd.DataFrame(
        [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
        index=al_inds,
    )

    #
    s_matrix.columns = list(bl_inds.get_level_values(level=-1))

    # now insert indicator for bottom level
    for i in s_matrix.columns:
        s_matrix.loc[s_matrix.index.get_level_values(-1) == i, i] = 1.0

    # now for each unique column
    for j in s_matrix.columns:

        # find bottom index id
        inds = list(s_matrix.index[s_matrix.index.get_level_values(level=-1).isin([j])])

        # generate new tuples for the aggregate levels
        for i in range(len(inds[0])):
            tmp = list(inds[i])
            tmp[-(i + 1)] = "__total"
            inds.append(tuple(tmp))

        # insrt indicator for aggregates
        for i in inds:
            s_matrix.loc[i, j] = 1.0

    # drop new levels not present in orginal matrix
    s_matrix.dropna(inplace=True)

    return s_matrix


# tests for matrix index?
def _get_g_matrix_bu(X):
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel("timepoints")
        .unique()
    )

    # get all level indexes
    al_inds = X.droplevel(level="timepoints").index.unique()

    g_matrix = pd.DataFrame(
        [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
        index=al_inds,
    )

    #
    g_matrix.columns = list(bl_inds.get_level_values(level=-1))

    # now insert indicator for bottom level
    for i in g_matrix.columns:
        g_matrix.loc[g_matrix.index.get_level_values(-1) == i, i] = 1.0

    return g_matrix.transpose()


def _get_g_matrix_ols(X):
    # get s matrix
    smat = _get_s_matrix(X)
    # get g
    g_ols = pd.DataFrame(
        np.dot(inv(np.dot(np.transpose(smat), smat)), np.transpose(smat))
    )
    # set indexes of matrix
    g_ols = g_ols.transpose()
    g_ols = g_ols.set_index(smat.index)
    g_ols.columns = smat.columns
    g_ols = g_ols.transpose()

    return g_ols


# this is similar to the ols except we have a new matrix W
# W is a matrix which simply counts the number of dissaggregated
# series connected to that node
# for quarterly temporal series W = diag(4, 2, 2, 1, 1, 1, 1)
def _get_g_matrix_wls_str(X):

    smat = _get_s_matrix(X)
    diag_data = np.diag(smat.sum(axis=1).values)
    w_mat = pd.DataFrame(diag_data, index=smat.index, columns=smat.index)

    g_wls_str = pd.DataFrame(
        np.dot(
            inv(np.dot(np.transpose(smat), np.dot(w_mat, smat))),
            np.dot(np.transpose(smat), w_mat),
        )
    )
    # set indexes of matrix
    g_wls_str = g_wls_str.transpose()
    g_wls_str = g_wls_str.set_index(smat.index)
    g_wls_str.columns = smat.columns
    g_wls_str = g_wls_str.transpose()

    return g_wls_str
