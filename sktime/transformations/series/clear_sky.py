# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Clear sky transformer for solar power forecasting."""

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd
from scipy.stats import vonmises
from statsmodels.stats.weightstats import DescrStatsW

from sktime.transformations.base import BaseTransformer

# todo: bayesian updating?
# todo: some optimisation of kernel bandwidths?

# next steps
# move estimate into fit
# clock changes
# unique combination of tod/doy as lookup table
# threshold as parameter


class ClearSky(BaseTransformer):
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

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
        ],  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.DatetimeIndex
        ],  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        "capability:unequal_length:removes": True,  # ?
        "handles-missing-data": False,
        "capability:missing_values:removes": True,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self, quantile_prob=0.95, bw_diurnal=100, bw_annual=10):

        self.quantile_prob = quantile_prob
        self.bw_diurnal = bw_diurnal
        self.bw_annual = bw_annual

        super(ClearSky, self).__init__()

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
        # get required seasonal aspects
        df = pd.DataFrame(index=X.index)
        df["yday"] = df.index.dayofyear
        df["tod"] = df.index.hour + df.index.minute / 60 + df.index.second / 60

        csp = df.apply(
            lambda x: _clearskypower(
                y=X,
                q=self.quantile_prob,
                tod_i=x.tod,
                doy_i=x.yday,
                tod_vec=df["tod"],
                doy_vec=df["yday"],
                bw_tod=self.bw_diurnal,
                bw_doy=self.bw_annual,
            ),
            axis=1,
        )

        X_trafo = X / csp

        X_trafo[(X / X.max()) < 0.01] = 0

        return X_trafo

    # # todo: consider implementing this, optional
    # # if not implementing, delete the _inverse_transform method
    # # inverse transform exists only if transform does not change scitype
    # #  i.e., Series transformed to Series
    # def _inverse_transform(self, X, y=None):
    #     """Inverse transform, inverse operation to transform.

    #     private _inverse_transform containing core logic, called from
    # inverse_transform

    #     Parameters
    #     ----------
    #     X : Series or Panel of mtype X_inner_mtype
    #         if X_inner_mtype is list, _inverse_transform must support all types in it
    #         Data to be inverse transformed
    #     y : Series or Panel of mtype y_inner_mtype, optional (default=None)
    #         Additional data, e.g., labels for transformation

    #     Returns
    #     -------
    #     inverse transformed version of X
    #     """
    #     # implement here
    #     # IMPORTANT: avoid side effects to X, y
    #     #
    #     # type conventions are exactly those in _transform, reversed
    #     #
    #     # for example: if transform-output is "Series":
    #     #  return should be of same mtype as input, X_inner_mtype
    #     #  if multiple X_inner_mtype are supported, ensure same input/output
    #     #
    #     # todo: add the return mtype/scitype to the docstring, e.g.,
    #     #  Returns
    #     #  -------
    #     #  X_inv_transformed : Series of mtype pd.DataFrame
    #     #       inverse transformed version of X

    # # todo: consider implementing this, optional
    # # if not implementing, delete the _update method
    # # standard behaviour is "no update"
    # # also delete in the case where there is no fitting
    # def _update(self, X, y=None):
    #     """Update transformer with X and y.

    #     private _update containing the core logic, called from update

    #     Parameters
    #     ----------
    #     X : Series or Panel of mtype X_inner_mtype
    #         if X_inner_mtype is list, _update must support all types in it
    #         Data to update transformer with
    #     y : Series or Panel of mtype y_inner_mtype, default=None
    #         Additional data, e.g., labels for tarnsformation

    #     Returns
    #     -------
    #     self: reference to self
    #     """
    #     # implement here
    #     # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
    #     # IMPORTANT: avoid side effects to X, y
    #     #
    #     # any model parameters should be written to attributes ending in "_"
    #     #  attributes set by the constructor must not be overwritten
    #     #  if used, estimators should be cloned to attributes ending in "_"
    #     #  the clones, not the originals, should be used or fitted if needed

    # # todo: consider implementing this, optional
    # # if not implementing, delete the method
    # def get_fitted_params(self):
    #     """Get fitted parameters.

    #     Returns
    #     -------
    #     fitted_params : dict
    #     """
    #     # implement here

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
        params = {"quantile_prob": 0.95, "bw_diurnal": 100, "bw_annual": 10}

        return params


def _clearskypower(y, q, tod_i, doy_i, tod_vec, doy_vec, bw_tod, bw_doy):
    """Docstring."""
    wts_tod = vonmises.pdf(
        x=tod_i * 2 * np.pi / 24, kappa=bw_tod, loc=tod_vec * 2 * np.pi / 24
    )
    wts_doy = vonmises.pdf(
        x=doy_i * 2 * np.pi / 365.25, kappa=bw_doy, loc=doy_vec * 2 * np.pi / 365.25
    )

    wts = wts_doy * wts_tod
    wts = wts / wts.sum()

    csp = DescrStatsW(y, weights=wts).quantile(probs=q).values[0]

    return csp
