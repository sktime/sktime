# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements reconciled forecasters for hierarchical data."""

__all__ = ["ReconcilerForecaster"]
__author__ = [
    "ciaran-g",
]

# todo: include the reconciler transformers?
# todo: top down historical proportions?

from warnings import warn

import numpy as np
import pandas as pd
from numpy.linalg import inv

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.transformations.hierarchical.aggregate import _check_index_no_total
from sktime.transformations.hierarchical.reconcile import (
    _get_s_matrix,
    _parent_child_df,
)


class ReconcilerForecaster(BaseForecaster):
    """Hierarchical reconcilation forecaster.

    Reconciliation is applied to make the forecasts in a hierarchy of
    time-series sum together appropriately.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    forecaster : estimator
        Estimator to generate base forecasts which are then reconciled
    method : {"mint_cov", "mint_shrink", "wls_var"}, default="mint_cov"
        The reconciliation approach applied to the forecasts based on
            "mint_cov" - sample covariance
            "mint_shrink" - covariance with shrinkage
            "wls_var" - weighted least squares (variance)

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """

    _required_parameters = ["forecaster"]

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "fit_is_empty": False,
    }

    METHOD_LIST = ["mint_cov", "mint_shrink", "wls_var"]

    def __init__(self, forecaster, method="mint_cov"):

        self.forecaster = forecaster
        self.method = method

        super(ReconcilerForecaster, self).__init__()

    def _add_totals(self, y):
        """Add total levels to y, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(y)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, default=None
            Exogenous variables for the base forecaster

        Returns
        -------
        self : reference to self
        """
        self._check_method()

        # # check the length of index if not hierarchical just return forecaster
        if y.index.nlevels < 2:
            self.forecaster_ = self.forecaster.clone()
            self.forecaster_.fit(y=y, X=X, fh=fh)
            return self

        # check index for no "__total", if not add totals to y
        if _check_index_no_total(y):
            y = self._add_totals(y)

        if X is not None:
            if _check_index_no_total(X):
                X = self._add_totals(X)

        # fit forecasters for each level
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        # now summation matrix
        self.s_matrix = _get_s_matrix(y)

        # parent child df
        self.parent_child = _parent_child_df(self.s_matrix)

        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        self.residuals = y - self.forecaster_.predict(fh=fh_resid, X=X)

        if self.method == "mint_cov":
            self.g_matrix = self._get_g_matrix_mint(shrink=False)
        elif self.method == "mint_shrink":
            self.g_matrix = self._get_g_matrix_mint(shrink=True)
        elif self.method == "wls_var":
            self.g_matrix = self._get_g_matrix_mint(shrink=False, diag_only=True)
        else:
            raise RuntimeError("unreachable condition, error in _check_method")

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        base_fc = self.forecaster_.predict(fh=fh, X=X)

        if base_fc.index.nlevels < 2:
            warn(
                "Reconciler is intended for use with y.index.nlevels > 1. "
                "Returning predictions unchanged."
            )
            return base_fc

        base_fc = base_fc.groupby(level=-1)

        recon_fc = []
        for _name, group in base_fc:
            # reconcile via SGy
            fcst = self.s_matrix.dot(self.g_matrix.dot(group.droplevel(-1)))
            # add back in time index
            fcst.index = group.index
            recon_fc.append(fcst)

        recon_fc = pd.concat(recon_fc, axis=0)
        recon_fc = recon_fc.sort_index()

        return recon_fc

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables based to the base forecaster
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if y.index.nlevels < 2:
            self.forecaster_.update(y, X, update_params=update_params)
            return self

        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        self.residuals = y - self.forecaster_.predict(fh=fh_resid, X=X)

        # could implement something specific here
        # for now just refit
        if self.method == "mint_cov":
            self.g_matrix = self._get_g_matrix_mint(shrink=False)
        elif self.method == "mint_shrink":
            self.g_matrix = self._get_g_matrix_mint(shrink=True)
        elif self.method == "wls_var":
            self.g_matrix = self._get_g_matrix_mint(shrink=False, diag_only=True)
        else:
            raise RuntimeError("unreachable condition, error in _check_method")

        return self

    def _get_g_matrix_mint(self, shrink=False, diag_only=False):
        """Define the G matrix for the MinT methods based on model residuals.

        Reconciliation methods require the G matrix. The G matrix is used to redefine
        base forecasts for the entire hierarchy to the bottom-level only before
        summation using the S matrix.

        Please refer to [1]_ for further information.

        Parameters
        ----------
        shrink:  bool, optional (default=False)
            Shrink the off diagonal elements of the sample covariance matrix.
            according to the method in [2]_
        diag_only: bool, optional (default=False)
            Remove the off-diagonal elements of the sample covariance matrix.

        Returns
        -------
        g_mint : pd.DataFrame with rows equal to the number of bottom level nodes
            only, i.e. with no aggregate nodes, and columns equal to the number of
            unique nodes in the hierarchy. The matrix indexes is inherited from the
            input data, with the time level removed.

        References
        ----------
        .. [1] https://otexts.com/fpp3/hierarchical.html
        .. [2] https://doi.org/10.2202/1544-6115.1175
        """
        if self.residuals.index.nlevels < 2:
            return None

        # copy for further mods
        resid = self.residuals.copy()
        # mean scale
        grp_range = np.arange(self.residuals.index.nlevels - 1).tolist()
        resid = resid.groupby(level=grp_range).apply(lambda x: x - x.mean())
        # cov matrix
        resid = resid.unstack().transpose()
        nobs = len(resid)
        cov_mat = resid.transpose().dot(resid) / (nobs - 1)

        if shrink:
            # diag matrix of variances
            var_d = pd.DataFrame(0.0, index=cov_mat.index, columns=cov_mat.columns)
            np.fill_diagonal(var_d.values, np.diag(cov_mat))

            # get correltion from covariance above
            cor_mat = (np.diag(cov_mat)) ** (-1 / 2)
            scale_m = pd.DataFrame(
                [cor_mat] * len(cor_mat), index=cov_mat.index, columns=cov_mat.columns
            )
            cor_mat = cov_mat * (scale_m) * (scale_m.transpose())

            # first scale the residuals
            resid_ho = resid.apply(lambda x: (x / x.std()))
            # scale for higher order cor
            scale_ho = ((resid_ho).transpose().dot((resid_ho))) ** 2 * (1 / nobs)
            resid_ho = resid_ho**2

            # higherorder cor (only diags)
            corho_mat = (resid_ho.transpose().dot(resid_ho)) - scale_ho
            corho_mat = (nobs / ((nobs - 1)) ** 3) * corho_mat

            # set diagonals to zero
            for i in resid.columns:
                corho_mat.loc[corho_mat.index == i, corho_mat.columns == i] = 0
                cor_mat.loc[cor_mat.index == i, cor_mat.columns == i] = 0

            # get the shrinkage value
            lamb = corho_mat.sum().sum() / (cor_mat**2).sum().sum()
            lamb = np.min([1, np.max([0, lamb])])

            # shrink the matrix
            cov_mat = (lamb * var_d) + ((1 - lamb) * cov_mat)

        if diag_only:
            # digonal matrix of variances
            for i in resid.columns:
                cov_mat.loc[cov_mat.index != i, cov_mat.columns == i] = 0

        # now get the g matrix based on the covariance
        g_mint = pd.DataFrame(
            np.dot(
                inv(
                    np.dot(np.transpose(self.s_matrix), np.dot(cov_mat, self.s_matrix))
                ),
                np.dot(np.transpose(self.s_matrix), cov_mat),
            )
        )
        # set indexes of matrix
        g_mint = g_mint.transpose()
        g_mint = g_mint.set_index(self.s_matrix.index)
        g_mint.columns = self.s_matrix.columns
        g_mint = g_mint.transpose()

        return g_mint

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
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing

        FORECASTER = ExponentialSmoothing()
        params_list = [
            {
                "forecaster": FORECASTER,
                "method": x,
            }
            for x in cls.METHOD_LIST
        ]
        return params_list
