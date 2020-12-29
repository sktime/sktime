#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

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
from scipy.stats.morestats import _boxcox_conf_interval
from scipy.stats.morestats import _calc_uniform_order_statistic_medians

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


# TODO replace with scipy version once PR for adding bounds is merged
def _boxcox_normmax(x, bounds=None, brack=(-2.0, 2.0), method="pearsonr"):
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

    def _pearsonr(x):
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            # This function computes the x-axis values of the probability plot
            # and computes a linear regression (including the correlation) and
            # returns ``1 - r`` so that a minimization function maximizes the
            # correlation.
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


def _boxcox(x, lmbda=None, bounds=None, alpha=None):
    r"""
    Return a dataset transformed by a Box-Cox power transformation.
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
    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')
    We now use `boxcox` to transform the data so it's closest to normal:
    >>> ax2 = fig.add_subplot(212)
    >>> xt, _ = stats._boxcox(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Box-Cox transformation')
    >>> plt.show()
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


class BoxCoxTransformer(_SeriesToSeriesTransformer):
    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(self, bounds=None, method="mle"):
        self.bounds = bounds
        self.method = method
        self.lambda_ = None
        super(BoxCoxTransformer, self).__init__()

    def fit(self, Z, X=None):
        z = check_series(Z, enforce_univariate=True)
        self.lambda_ = _boxcox_normmax(z, bounds=self.bounds, method=self.method)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        zt = boxcox(z.to_numpy(), self.lambda_)
        return pd.Series(zt, index=z.index)

    def inverse_transform(self, Z, X=None):
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
