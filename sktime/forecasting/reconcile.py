# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements reconciled forecasters for hierarchical data."""

__all__ = ["ReconcilerForecaster"]
__author__ = [
    "ciaran-g",
]

# check adding "_" suffix to returned items self.s_matrix_?
# check the predict residuals bug
# include the default vectorized fit tag properly
# include the shrinkage, wls_var, topdown_prop estimators
# documentation
# set up the tests
# include the reconciler transformers
# check against fable?

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
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": "None",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
    }

    # METHOD_LIST = ["bu", "ols", "wls_str", "td_fcst"]
    METHOD_LIST = ["mint"]

    def __init__(self, forecaster, method="mint"):

        self.forecaster = forecaster
        self.method = method

        super(ReconcilerForecaster, self).__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "ignores-exogeneous-X",
            "handles-missing-data",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)
        # don't fit the reconciler vectorized
        self.default_vectorized = False

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
        # get the original from vectorizedDF (baseclass)
        y = self._y.reconstruct(self._y)
        self._check_method()

        # # check the length of index if not hierarchical just return forecaster
        if y.index.nlevels < 2:
            self.forecaster_ = self.forecaster.clone()
            self.forecaster_.fit(y=y, X=X, fh=fh)
            return self

        # what about aggregation of exogenous variables (X) here...
        # # hmm maybe just sum again? discuss
        # check index for no "__total", if not add totals to y
        if _check_index_no_total(y):
            y = self._add_totals(y)

        # fit forecasters for each level
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        # from vectorized to original df again
        y = self._y.reconstruct(self._y)
        # now summation matrix
        self.s_matrix = _get_s_matrix(y)

        # parent child df
        self.parent_child_ = _parent_child_df(self.s_matrix)

        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        resid = y - self.forecaster_.predict(fh=fh_resid, X=X)
        # scale
        grp_range = np.arange(resid.index.nlevels - 1).tolist()
        resid = resid.groupby(level=grp_range).apply(lambda x: x - x.mean())

        self.scaled_residual_df_ = resid.unstack().transpose()

        if self.method == "mint":
            self.g_matrix_ = self._get_g_matrix_mint(shrink=False)
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
        base_fc = base_fc.groupby(level=-1)

        recon_fc = []
        for _name, group in base_fc:
            # reconcile via SGy
            fcst = self.s_matrix.dot(self.g_matrix_.dot(group.droplevel(-1)))
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
        self.forecaster_.update(y, X, update_params=update_params)

        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        resid = y - self.forecaster_.predict(fh=fh_resid, X=X)
        # scale
        grp_range = np.arange(resid.index.nlevels - 1).tolist()
        resid = resid.groupby(level=grp_range).apply(lambda x: x - x.mean())

        self.scaled_residual_df_ = resid.unstack().transpose()

        if self.method == "mint":
            # could implement something specific here
            # for now just refit
            self.g_matrix_ = self._get_g_matrix_mint(shrink=False)
        else:
            raise RuntimeError("unreachable condition, error in _check_method")

        return self

    def _get_g_matrix_mint(self, shrink=False):
        """Define the G matrix for the MinT method."""
        if not shrink:
            cov_mat = self.scaled_residual_df_.transpose().dot(
                self.scaled_residual_df_
            ) / (len(self.scaled_residual_df_.index) - 1)
        else:
            pass

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
        params_list = [{"forecaster": FORECASTER, "method": x} for x in cls.METHOD_LIST]

        return params_list
