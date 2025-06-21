# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements hierarchical reconciliation transformers.

These reconcilers only depend on the structure of the hierarchy.
"""

__author__ = ["ciaran-g", "eenticott-shell", "k1m190r"]

import numpy as np
import pandas as pd
from numpy.linalg import inv

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import _check_index_no_total
from sktime.utils.warnings import warn

# TODO: failing test which are escaped


class Reconciler(BaseTransformer):
    """Hierarchical reconciliation transformer.

    Hierarchical reconciliation is a transformation which is used to make the
    predictions in a hierarchy of time-series sum together appropriately.

    The methods implemented in this class only require the structure of the
    hierarchy or the forecasts values for reconciliation.

    These functions are intended for transforming hierarchical forecasts, i.e.
    after prediction. However they are general and can be used to transform
    hierarchical time-series data.

    For reconciliation methods that require historical values in addition to the
    forecasts, such as MinT, see the ``ReconcilerForecaster`` class.

    For more versatile and efficient reconciliation in pipelines,
    see ``BottomUpReconciler``,
    ``TopdownReconciler``, ``OptimalReconciler``,
    ``NonNegativeOptimalReconciler``, ``MiddleOutReconciler``, that apply
    reconciliation as preprocessing and postprocessing steps.

    For further information on the methods, see [1]_.

    Parameters
    ----------
    method : {"bu", "ols", "wls_str", "td_fcst"}, default="bu"
        The reconciliation approach applied to the forecasts

        * ``"bu"`` - bottom-up
        * ``"ols"`` - ordinary least squares
        * ``"wls_str"`` - weighted least squares (structural)
        * ``"td_fcst"`` - top down based on (forecast) proportions

    See Also
    --------
    Aggregator
    ReconcilerForecaster
    BottomUpReconciler
    TopdownReconciler
    OptimalReconciler
    NonNegativeOptimalReconciler
    MiddleOutReconciler

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html

    Examples
    --------
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.transformations.hierarchical.reconcile import Reconciler
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> agg = Aggregator()
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=3,
    ...     no_levels=1,
    ...     random_seed=123,
    ... )
    >>> y = agg.fit_transform(y)
    >>> forecaster = PolynomialTrendForecaster()
    >>> forecaster.fit(y)
    PolynomialTrendForecaster(...)
    >>> prds = forecaster.predict(fh=[1])
    >>> # reconcile forecasts
    >>> reconciler = Reconciler(method="ols")
    >>> prds_recon = reconciler.fit_transform(prds)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ciaran-g", "eenticott-shell", "k1m190r"],
        "maintainers": "ciaran-g",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    METHOD_LIST = ["bu", "ols", "wls_str", "td_fcst"]

    def __init__(self, method="bu"):
        self.method = method

        super().__init__()

    def _add_totals(self, X):
        """Add total levels to X, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(X)

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
        self._check_method()

        # check the length of index
        if X.index.nlevels < 2:
            return self

        # check index for no "__total", if not add totals to X
        if _check_index_no_total(X):
            X = self._add_totals(X)

        # define reconciliation matrix
        if self.method == "bu":
            self.g_matrix = _get_g_matrix_bu(X)
        elif self.method == "ols":
            self.g_matrix = _get_g_matrix_ols(X)
        elif self.method == "wls_str":
            self.g_matrix = _get_g_matrix_wls_str(X)
        elif self.method == "td_fcst":
            self.g_matrix = _get_g_matrix_td_fcst(X)
        else:
            raise RuntimeError("unreachable condition, error in _check_method")

        # now summation matrix
        self.s_matrix = _get_s_matrix(X)

        # parent child df
        self.parent_child = _parent_child_df(self.s_matrix)

        return self

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
        # check the length of index
        if X.index.nlevels < 2:
            warn(
                "Reconciler is intended for use with X.index.nlevels > 1. "
                "Returning X unchanged.",
                obj=self,
            )
            return X

        # check index for no "__total", if not add totals to X
        if _check_index_no_total(X):
            warn(
                "No elements of the index of X named '__total' found. Adding "
                "aggregate levels using the default Aggregator transformer "
                "before reconciliation.",
                obj=self,
            )
            X = self._add_totals(X)

        # check here that index of X matches the self.s_matrix
        al_inds = X.droplevel(level=-1).index.unique()
        chk_newindx = np.all(self.s_matrix.index == al_inds)
        if not chk_newindx:
            raise ValueError(
                "Check unique indexes of X.droplevel(level=-1) matches "
                "the data used in Reconciler().fit(X)."
            )

        X = X.groupby(level=-1)
        # could use X.transform() with np.dot, v. marginally faster in my tests
        # - loop can use index matching via df.dot() which is probably worth it
        recon_preds = []
        gmat = self.g_matrix
        for _name, group in X:
            if self.method == "td_fcst":
                gmat = _update_td_fcst(
                    g_matrix=gmat, x_sf=group.droplevel(-1), conn_df=self.parent_child
                )
            # reconcile via SGy
            fcst = self.s_matrix.dot(gmat.dot(group.droplevel(-1)))
            # add back in time index
            fcst.index = group.index
            recon_preds.append(fcst)

        recon_preds = pd.concat(recon_preds, axis=0)
        recon_preds = recon_preds.sort_index()

        return recon_preds

    def _check_method(self):
        """Raise warning if method is not defined correctly."""
        if not np.isin(self.method, self.METHOD_LIST):
            raise ValueError(f"""method must be one of {self.METHOD_LIST}.""")
        else:
            pass

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{"method": x} for x in cls.METHOD_LIST]


def _get_s_matrix(X):
    """Determine the summation "S" matrix.

    Reconciliation methods require the S matrix, which is defined by the
    structure of the hierarchy only. The S matrix is inferred from the input
    multi-index of the forecasts and is used to sum bottom-level forecasts
    appropriately.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    s_matrix : pd.DataFrame with rows equal to the number of unique nodes in
        the hierarchy, and columns equal to the number of bottom level nodes only,
        i.e. with no aggregate nodes. The matrix indexes is inherited from the
        input data, with the time level removed.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel(level=-1)
        .unique()
    )
    # get all level indexes
    al_inds = X.droplevel(level=-1).index.unique()

    # set up matrix
    s_matrix = pd.DataFrame(
        [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
        index=al_inds,
    )
    s_matrix.columns = bl_inds

    # now insert indicator for bottom level
    for i in s_matrix.columns:
        s_matrix.loc[s_matrix.index == i, i] = 1.0

    # now for each unique column add aggregate indicator
    for i in s_matrix.columns:
        if s_matrix.index.nlevels > 1:
            # replace index with totals -> ("nodeA", "__total")
            agg_ind = list(i)[::-1]
            for j in range(len(agg_ind)):
                agg_ind[j] = "__total"
                # insert indicator
                s_matrix.loc[tuple(agg_ind[::-1]), i] = 1.0
        else:
            s_matrix.loc["__total", i] = 1.0

    # drop new levels not present in original matrix
    s_matrix = s_matrix.loc[s_matrix.index.isin(al_inds)]

    return s_matrix


def _get_g_matrix_bu(X):
    """Determine the reconciliation "G" matrix for the bottom up method.

    Reconciliation methods require the G matrix. The G matrix is used to redefine
    base forecasts for the entire hierarchy to the bottom-level only before
    summation using the S matrix.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    g_matrix : pd.DataFrame with rows equal to the number of bottom level nodes
        only, i.e. with no aggregate nodes, and columns equal to the number of
        unique nodes in the hierarchy. The matrix indexes is inherited from the
        input data, with the time level removed.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
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
    g_matrix.columns = bl_inds

    # now insert indicator for bottom level
    for i in g_matrix.columns:
        g_matrix.loc[g_matrix.index == i, i] = 1.0

    return g_matrix.transpose()


def _get_g_matrix_ols(X):
    """Determine the reconciliation "G" matrix for the ordinary least squares method.

    Reconciliation methods require the G matrix. The G matrix is used to redefine
    base forecasts for the entire hierarchy to the bottom-level only before
    summation using the S matrix.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    g_ols : pd.DataFrame with rows equal to the number of bottom level nodes
        only, i.e. with no aggregate nodes, and columns equal to the number of
        unique nodes in the hierarchy. The matrix indexes is inherited from the
        summation matrix.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
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
    """Reconciliation "G" matrix for the weighted least squares (structural) method.

    Reconciliation methods require the G matrix. The G matrix is used to re-define
    base forecasts for the entire hierarchy to the bottom-level only before
    summation using the S matrix.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    g_wls_str : pd.DataFrame with rows equal to the number of bottom level nodes
        only, i.e. with no aggregate nodes, and columns equal to the number of
        unique nodes in the hierarchy. The matrix indexes is inherited from the
        summation matrix.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
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


def _get_g_matrix_td_fcst(X):
    """Determine the "G" matrix for the top down forecast proportions method.

    Reconciliation methods require the G matrix. The G matrix is used to redefine
    base forecasts for the entire hierarchy to the bottom-level only before
    summation using the S matrix.

    Note that the G matrix for this method changes for each forecast. This
    is just a template G matrix which is updated at each iteration.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    g_matrix : pd.DataFrame with rows equal to the number of bottom level nodes
        only, i.e. with no aggregate nodes, and columns equal to the number of
        unique nodes in the hierarchy. The matrix indexes is inherited from the
        input data, with the time level removed.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """
    g_matrix = _get_g_matrix_bu(X)
    g_matrix = g_matrix.replace(to_replace=1, value=0)

    return g_matrix


def _update_td_fcst(g_matrix, x_sf, conn_df):
    """Update the "G" matrix for the top down forecast proportions method.

    Reconciliation methods require the G matrix. The G matrix is used to redefine
    base forecasts for the entire hierarchy to the bottom-level only before
    summation using the S matrix.

    This takes the gmatrix template from _get_g_matrix_td_fcst() and updates
    it based on a single forecast.
    Please refer to [1]_ for further information.

    Parameters
    ----------
    g_matrix :  pd.DataFrame reconciliation matrix template from
        _get_g_matrix_td_fcst()
    x_sf : pd.Series which contains a hierarchy forecast for a single timepoint
    conn_df : A look up table containing the child and parent of each connection
        in a hierarchy

    Returns
    -------
    g_matrix : pd.DataFrame with rows equal to the number of bottom level nodes
        only, i.e. with no aggregate nodes, and columns equal to the number of
        unique nodes in the hierarchy. The matrix indexes is inherited from the
        input data, with the time level removed.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """
    for i in g_matrix.index:
        # start from each bottom index
        child = i
        props = []
        # if the bottom level are single strings, integers, or whatever
        if not isinstance(child, tuple):
            child_chk = (child,)
        else:
            child_chk = child

        while sum([j == "__total" for j in list(child_chk)]) < len(child_chk):
            # find the parent of the child
            parent = conn_df.loc[conn_df["child"] == child, "parent"].values[0]
            # now need to find nodes directly connected to the parent
            children = conn_df.loc[conn_df["parent"] == parent, "child"].unique()
            # calculate proportions
            props.append((x_sf.loc[child] / x_sf.loc[children].sum()).values[0])
            # move up the chain
            child = parent
            if not isinstance(child, tuple):
                child_chk = (child,)
            else:
                child_chk = child

        g_matrix.loc[i, "__total"] = np.prod(props)

    return g_matrix


def _parent_child_df(s_matrix):
    """Extract the parent and child connections in a hierarchy.

    This function takes the summation S matrix for a given hierarchy and
    returns a dataframe containing a the parent and child node id for each
    connection in a hierarchy.

    Parameters
    ----------
    s_matrix :  The summation matrix for a given hierarchy from the function
        _get_s_matrix().

    Returns
    -------
    df : A two column pd.DataFrame with rows equal to the number of
        connections in a hierarchy.
    """
    parent_child = []
    total_count = s_matrix.index.to_frame()
    total_count = (total_count == "__total").sum(axis=1)

    # for each bottom node
    for i in s_matrix.columns:
        # get all connections
        connected_nodes = s_matrix[(s_matrix[i] == 1)].sum(axis=1)
        # for non-flattened hierarchies make sure "__totals" are above
        connected_nodes = (connected_nodes + total_count).dropna()
        connected_nodes = connected_nodes.sort_values(ascending=False)

        # starting from top add list of [parent, child]
        for j in range(len(connected_nodes.index) - 1):
            parent_child.append(
                [connected_nodes.index[j], connected_nodes.index[j + 1]]
            )

    df = pd.DataFrame(parent_child)
    df.columns = ["parent", "child"]

    df = df.drop_duplicates().sort_values(["parent", "child"]).reset_index(drop=True)

    return df
