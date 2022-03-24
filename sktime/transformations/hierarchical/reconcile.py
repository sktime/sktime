# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements hierarchical reconciliation transformers.

These reconcilers only depend on the structure of the hierarchy.
"""

__author__ = ["ciaran-g", "eenticott-shell", "k1m190r"]

import numpy as np
import pandas as pd
from numpy.linalg import inv

from sktime.transformations.base import BaseTransformer


class reconciler(BaseTransformer):
    """Hierarchical reconcilation transformer.

    Hierarchical reconciliation is a transfromation which is used to make the
    predictions in a hierarchy of time-series sum together appropriately.

    The methods implemented in this class only require the structure of the
    hierarchy to reconcile the forecasts.

    Please refer to [1]_ for further information

    Parameters
    ----------
    method : {"bu", "ols", "wls_str"}, default="ols"
        The reconciliation approach applied to the forecasts
            "bu" - bottom-up
            "ols" - ordinary least squares
            "wls_str" - weighted least squares (structural)

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "pd_multiindex_hier",
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
    }

    # test that method is recognised
    def __init__(self, method="bu"):

        self.method = method

        super(reconciler, self).__init__()

    # test for type of input?
    # test for __total present in index?
    # tests for index matching at each time point?
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Panel of mtype pd_multiindex_hier
            Data to fit transform to
        y :  Ignored argument for interface compatibility.

        Returns
        -------
        self: reference to self
        """
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
        X : Panel of mtype pd_multiindex_hier
            Data to be transformed
        y : Ignored argument for interface compatibility.

        Returns
        -------
        recon_preds : multi-indexed pd.DataFrame of Panel mtype pd_multiindex
        """
        # include index between matrices here as in df.dot()?
        X = X.groupby(level=-1)
        recon_preds = X.transform(
            lambda y: np.dot(self.s_matrix, np.dot(self.g_matrix, y))
        )

        return recon_preds

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"method": "bu"}

        return params


# tests for matrix index?
# what happens if two end-points have the same name?
def _get_s_matrix(X):
    """Determine the summation "S" matrix.

    Returns
    -------
    s_matrix : pd.DataFrame
        Summation matrix with multiindex on rows, where each row is a level in
        the hierarchy. Single index on columns for each bottom level of the
        hierarchy.
    """
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel(level=-1)
        .unique()
    )
    # get all level indexes
    al_inds = X.droplevel(level=-1).index.unique()

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
    """Determine the reconciliation "G" matrix for the bottom up method.

    Returns
    -------
    g_matrix : pd.DataFrame
        Summation matrix with single index on rows and multiindex on columns.
    """
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel(level=-1)
        .unique()
    )

    # get all level indexes
    al_inds = X.droplevel(level=-1).index.unique()

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
    """Determine the reconciliation "G" matrix for the ols method.

    Returns
    -------
    g_ols : pd.DataFrame
        Summation matrix with single index on rows and multiindex on columns.
    """
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


def _get_g_matrix_wls_str(X):
    """Determine the reconciliation "G" matrix for the wls_str method.

    Returns
    -------
    g_wls_str : pd.DataFrame
        Summation matrix with single index on rows and multiindex on columns.
    """
    # this is similar to the ols except we have a new matrix W
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
