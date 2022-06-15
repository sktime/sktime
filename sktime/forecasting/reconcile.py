# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements reconciled forecasters for hierarchical data."""

__all__ = ["ReconcilerForecaster"]
__author__ = [
    "ciaran-g",
]

# check shrinkage reconciler
# https://strimmerlab.github.io/publications/journals/shrinkcov2005.pdf
# looks like the shrinkage estimate needs another look...
# documentation

# include the reconciler transformers?
# todo: top down historical proportions


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
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

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

    _required_parameters = ["forecaster"]

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
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

    def __init__(self, forecaster, method="mint_cov", mean_scale_residuals=True):

        self.forecaster = forecaster
        self.method = method
        self.mean_scale_residuals = mean_scale_residuals

        super(ReconcilerForecaster, self).__init__()

        # tags_to_clone = [
        #     "requires-fh-in-fit",
        #     "ignores-exogeneous-X",
        #     "X-y-must-have-same-index",
        #     "enforce_index_type",
        # ]
        # self.clone_tags(self.forecaster, tags_to_clone)
        # don't fit the reconciler vectorized
        # self.default_vectorized = False

    def _add_totals(self, y):
        """Add total levels to y, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(y)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

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

        if self.mean_scale_residuals:
            grp_range = np.arange(self.residuals.index.nlevels - 1).tolist()
            self.residuals = self.residuals.groupby(level=grp_range).apply(
                lambda x: x - x.mean()
            )

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

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
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

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
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

        if self.mean_scale_residuals:
            grp_range = np.arange(self.residuals.index.nlevels - 1).tolist()
            self.residuals = self.residuals.groupby(level=grp_range).apply(
                lambda x: x - x.mean()
            )

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
        """Define the G matrix for the MinT method."""
        if self.residuals.index.nlevels < 2:
            return None

        # copy in case of further mods?
        resid = self.residuals.copy()
        # cov matrix
        resid = resid.unstack().transpose()
        nobs = len(resid)
        cov_mat = resid.transpose().dot(resid) / nobs

        # shrink method of https://doi.org/10.2202/1544-6115.1175
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

            # scale the reiduals by the variance
            for i in resid.columns:
                scale = scale_m.loc[scale_m.index == i, scale_m.columns == i].values[0]
                resid[i] = resid[i] * scale

            crossp = (resid).transpose().dot((resid))
            crossp2 = (resid**2).transpose().dot((resid**2))

            v = (1 / (nobs * (nobs - 1))) * (crossp2 - (1 / (nobs * (crossp) ** 2)))
            # set diagonals to zero
            for i in resid.columns:
                v.loc[v.index == i, v.columns == i] = 0
                cor_mat.loc[cor_mat.index == i, cor_mat.columns == i] = 0

            d = cor_mat**2

            # get the shrinkage value
            lamb = v.sum().sum() / d.sum().sum()
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
                "mean_scale_residuals": True,
            }
            for x in cls.METHOD_LIST
        ]

        params_list = params_list + [
            {
                "forecaster": FORECASTER,
                "method": x,
                "mean_scale_residuals": False,
            }
            for x in cls.METHOD_LIST
        ]

        return params_list
