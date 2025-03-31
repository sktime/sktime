# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements hierarchical reconciliation transformers.

These reconcilers only depend on the structure of the hierarchy.
"""

__author__ = ["ciaran-g", "eenticott-shell", "k1m190r"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import _check_index_no_total
from sktime.transformations.hierarchical.reconciliation._utils import _loc_series_idxs
from sktime.transformations.hierarchical.reconciliation.bottom_up import (
    BottomUpReconciler,
)
from sktime.transformations.hierarchical.reconciliation.optimal import (
    NonNegativeOptimalReconciler,
    OptimalReconciler,
)
from sktime.transformations.hierarchical.reconciliation.topdown import (
    TopdownReconciler,
)
from sktime.utils.warnings import warn

__all__ = ["ReconcileForecasts"]


class ReconcileForecasts(BaseTransformer):
    """Hierarchical reconciliation transformer.

    Hierarchical reconciliation is a transformation which is used to make the
    predictions in a hierarchy of time-series sum together appropriately.

    The methods implemented in this class only require the structure of the
    hierarchy or the forecasts values for reconciliation.

    These functions are intended for transforming hierarchical forecasts, i.e.
    after prediction. If you are looking to transform the data before
    forecasting, please refer to BottomUpReconciler, OptimalReconciler,
    TopdownReconciler, or MiddleOutReconciler.

    For reconciliation methods that require historical values in addition to the
    forecasts, such as MinT, see the ``ReconcilerForecaster`` class.

    For further information on the methods, see [1]_.


    Parameters
    ----------
    method : {"bu", "ols", "wls_str", "td_fcst"}, default="bu"
        The reconciliation approach applied to the forecasts
            "bu" - bottom-up
            "ols" - ordinary least squares
            "wls_str" - weighted least squares (structural)
            "td_fcst" - top down based on (forecast) proportions

    See Also
    --------
    Aggregator
    ReconcilerForecaster

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
        "authors": ["ciaran-g", "eenticott-shell", "k1m190r", "felipeangelimvieira"],
        "maintainers": ["felipeangelimvieira", "ciaran-g"],
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
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
    }

    METHOD_LIST = ["bu", "ols", "ols:nonneg", "wls_str", "wls_str:nonneg", "td_fcst"]

    def __init__(self, method="bu"):
        self.method = method

        super().__init__()

    def _get_reconciler(self, X):
        if self.method == "bu":
            return BottomUpReconciler()
        # OLS
        elif self.method == "ols":
            return OptimalReconciler()
        elif self.method == "ols:nonneg":
            return NonNegativeOptimalReconciler("ols")
        # Structural WLS
        elif self.method == "wls_str":
            return OptimalReconciler("wls_str")
        elif self.method == "wls_str:nonneg":
            return NonNegativeOptimalReconciler("wls_str")
        elif self.method == "td_fcst":
            return TopdownReconciler()
        else:
            raise ValueError(f"""method must be one of {self.METHOD_LIST}.""")

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

        self._original_series = X.index.droplevel(-1).unique()
        self.reconciler_ = self._get_reconciler(X)
        self.reconciler_.fit(X)

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
            X_totals = self._add_totals(X)
            X = pd.concat(
                [
                    X_totals.loc[X_totals.index.difference(X.index)],
                    X,
                ],
                axis=0,
            ).sort_index()

        recon_preds = self.reconciler_.inverse_transform(X)

        recon_preds = _loc_series_idxs(recon_preds, self._original_series).sort_index()

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
        return [{"method": x} for x in cls.METHOD_LIST if not x.endswith("nonneg")]


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
