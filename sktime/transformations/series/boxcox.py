#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Markus Löning"]
__all__ = ["BoxCoxTransformer"]

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import special
from scipy import stats
from scipy.special import boxcox
from scipy.special import inv_boxcox
from scipy.stats import boxcox_llf
from scipy.stats import distributions
from scipy.stats import variation
from scipy.stats.morestats import _boxcox_conf_interval
from scipy.stats.morestats import _calc_uniform_order_statistic_medians

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation import is_int
from sktime.utils.validation.series import check_series


class BoxCoxTransformer(_SeriesToSeriesTransformer):
    """Box-Cox power transform.

    Example
    -------
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = BoxCoxTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(self, bounds=None, method="mle", sp=None):
        self.bounds = bounds
        self.method = method
        self.lambda_ = None
        self.sp = sp
        super(BoxCoxTransformer, self).__init__()

    def fit(self, Z, X=None):
        """Fit data.

        Parameters
        ----------
        Z : pd.Series
            Series to fit.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        self
        """
        z = check_series(Z, enforce_univariate=True)
        if self.method != "guerrero":
            self.lambda_ = _boxcox_normmax(z, bounds=self.bounds, method=self.method)
        else:
            self.lambda_ = _guerrero(z, self.sp, self.bounds)

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series
            Transformed series.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        zt = boxcox(z.to_numpy(), self.lambda_)
        return pd.Series(zt, index=z.index)

    def inverse_transform(self, Z, X=None):
        """Inverse transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series
            Transformed data - the inverse of the Box-Cox transformation.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        zt = inv_boxcox(z.to_numpy(), self.lambda_)
        return pd.Series(zt, index=z.index)


class LogTransformer(_SeriesToSeriesTransformer):
    _tags = {"transform-returns-same-time-index": True}

    def transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.log(Z)

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.exp(Z)


def _make_boxcox_optimizer(bounds=None, brack=(-2.0, 2.0)):
    # bounds is None, use simple Brent optimisation
    if bounds is None:

        def optimizer(func, args):
            return optimize.brent(func, brack=brack, args=args)

    # otherwise use bounded Brent optimisation
    else:
        # input checks on bounds
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f"`bounds` must be a tuple of length 2, but found: {bounds}"
            )

        def optimizer(func, args):
            return optimize.fminbound(func, bounds[0], bounds[1], args=args)

    return optimizer


# TODO replace with scipy version once PR for adding bounds is merged
def _boxcox_normmax(x, bounds=None, brack=(-2.0, 2.0), method="pearsonr"):
    optimizer = _make_boxcox_optimizer(bounds, brack)

    def _pearsonr(x):
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            y = _boxcox(samps, lmbda)
            yvals = np.sort(y)
            r, prob = stats.pearsonr(xvals, yvals)
            return 1 - r

        return optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):
        def _eval_mle(lmb, data):
            # function to minimize
            return -boxcox_llf(lmb, data)

        return optimizer(_eval_mle, args=(x,))

    def _all(x):
        maxlog = np.zeros(2, dtype=float)
        maxlog[0] = _pearsonr(x)
        maxlog[1] = _mle(x)
        return maxlog

    methods = {"pearsonr": _pearsonr, "mle": _mle, "all": _all}
    if method not in methods.keys():
        raise ValueError("Method %s not recognized." % method)

    optimfunc = methods[method]
    return optimfunc(x)


def _guerrero(x, sp, bounds=None):
    r"""Return lambda estimated by the Guerrero method [Guerrero].

    Parameters
    ----------
    x : ndarray
        Input array. Must be 1-dimensional.
    sp : integer
        Seasonal periodicity value. Must be an integer >= 2.
    bounds : {None, (float, float)}, optional
        Bounds on lambda to be used in minimization.

    Returns
    -------
    lambda : float
        Lambda value that minimizes the coefficient of variation of
        variances of the time series in different periods after
        Box-Cox transformation [Guerrero].

    References
    ----------
    [Guerrero] V.M. Guerrero, "Time-series analysis supported by Power
    Transformations ", Journal of Forecasting, vol. 12, pp. 37-48, 1993.
    """
    if sp is None or not is_int(sp) or sp < 2:
        raise ValueError(
            "Guerrero method requires an integer seasonal periodicity (sp) value >= 2."
        )

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")

    num_obs = len(x)
    len_prefix = num_obs % sp

    x_trimmed = x[len_prefix:]
    x_mat = x_trimmed.reshape((-1, sp))
    x_mean = np.mean(x_mat, axis=1)

    # [Guerrero, Eq.(5)] uses an unbiased estimation for
    # the standard deviation
    x_std = np.std(x_mat, axis=1, ddof=1)

    def _eval_guerrero(lmb, x_std, x_mean):
        x_ratio = x_std / x_mean ** (1 - lmb)
        x_ratio_cv = variation(x_ratio)
        return x_ratio_cv

    optimizer = _make_boxcox_optimizer(bounds)
    return optimizer(_eval_guerrero, args=(x_std, x_mean))


def _boxcox(x, lmbda=None, bounds=None, alpha=None):
    r"""Return a dataset transformed by a Box-Cox power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Must be positive 1-dimensional.  Must not be constant.
    lmbda : {None, scalar}, optional
        If `lmbda` is not None, do the transformation for that value.
        If `lmbda` is None, find the lambda that maximizes the log-likelihood
        function and return it as the second output argument.
    alpha : {None, float}, optional
        If ``alpha`` is not None, return the ``100 * (1-alpha)%`` confidence
        interval for `lmbda` as the third output argument.
        Must be between 0.0 and 1.0.

    Returns
    -------
    boxcox : ndarray
        Box-Cox power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.
    (min_ci, max_ci) : tuple of float, optional
        If `lmbda` parameter is None and ``alpha`` is not None, this returned
        tuple of floats represents the minimum and maximum confidence limits
        given ``alpha``.

    See Also
    --------
    probplot, boxcox_normplot, boxcox_normmax, boxcox_llf

    Notes
    -----
    The Box-Cox transform is given by::
        y = (x**lmbda - 1) / lmbda,  for lmbda > 0
            log(x),                  for lmbda = 0
    `boxcox` requires the input data to be positive.  Sometimes a Box-Cox
    transformation provides a shift parameter to achieve this; `boxcox` does
    not.  Such a shift parameter is equivalent to adding a positive constant to
    `x` before calling `boxcox`.
    The confidence limits returned when ``alpha`` is provided give the interval
    where:
    .. math::
        llf(\hat{\lambda}) - llf(\lambda) < \frac{1}{2}\chi^2(1 - \alpha, 1),
    with ``llf`` the log-likelihood function and :math:`\chi^2` the chi-squared
    function.

    References
    ----------
    G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
    Royal Statistical Society B, 26, 211-252 (1964).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")

    if x.size == 0:
        return x

    if np.all(x == x[0]):
        raise ValueError("Data must not be constant.")

    if any(x <= 0):
        raise ValueError("Data must be positive.")

    if lmbda is not None:  # single transformation
        return special.boxcox(x, lmbda)

    # If lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = _boxcox_normmax(x, bounds=bounds, method="mle")
    y = _boxcox(x, lmax)

    if alpha is None:
        return y, lmax
    else:
        # Find confidence interval
        interval = _boxcox_conf_interval(x, lmax, alpha)
        return y, lmax, interval
