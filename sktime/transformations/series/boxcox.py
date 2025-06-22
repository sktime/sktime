"""Implemenents Box-Cox and Log Transformations."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["mloning", "aiwalter", "fkiraly"]
__all__ = ["BoxCoxTransformer", "LogTransformer"]

import numpy as np

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation import is_int


# copy-pasted from scipy 1.7.3 since it moved in 1.8.0 and broke this estimator
# todo: find a suitable replacement
def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.
    """
    v = np.empty(n, dtype=np.float64)
    v[-1] = 0.5 ** (1.0 / n)
    v[0] = 1 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v


def _box_norm(X, bounds, method):
    """Adapter for boxcox_normmax pre and post scipy 1.7.0."""
    if _check_soft_dependencies("scipy<1.7.0", severity="none"):
        box_norm = _boxcox_normmax
        args = {"bounds": bounds}
    else:
        from scipy import optimize
        from scipy.stats import boxcox_normmax

        options = {"xatol": 1e-12}

        def optimizer(fun):
            return optimize.minimize_scalar(
                fun, bounds=bounds, method="bounded", options=options
            )

        box_norm = boxcox_normmax

        if bounds is not None:
            args = {"optimizer": optimizer}
        else:
            args = {"brack": bounds}

    return box_norm(X, method=method, **args)


class BoxCoxTransformer(BaseTransformer):
    r"""Box-Cox power transform.

    Box-Cox transformation is a power transformation that is used to
    make data more normally distributed and stabilize its variance based
    on the hyperparameter lambda. [1]_

    This transformer applies the Box-Cox-transform elementwise, where the lambda
    parameter is fitted by the specified method via ``method``.

    The Box-Cox-transform is defined as
    :math:`y\mapsto \frac{y^{\lambda}-1}{\lambda}, \lambda
    \ne 0 \text{ and } ln(y), \lambda = 0`,
    for positive :math:`y`.

    The :math:`\lambda` parameter is fitted per time series and instance and variable,
    by a method depending on the ``method`` parameter:

    * ``"pearsonr"`` - maximization of Pearson correlation between transformed
      and normalized untransformed. Direct interface to ``scipy.stats.boxcox_normmax``
      with ``method="pearsonr"``, with ``bracket=bounds``,  and otherwise defaults.
    * ``"mle"`` - maximization of the Box-Cox log-likelihood.
      Direct interface to ``scipy.stats.boxcox_normmax`` with ``method="mle"``
      with ``bracket=bounds``, and otherwise defaults.
    * ``"guerrero"`` - Guerrero's method with seasonal periodicity, see [2]_.
      this requires the seasonality parameter to be passed as ``sp``.
    * ``"fixed"`` - fixed, pre-specified :math:`\lambda`,
      which is passed as ``lambda_fixed``.

    If non-positive ``:math:y`` are present, they are by default replaced with their
    absolute values in ``fit``.
    In ``transform``, the signed Box-Cox-transform is applied, i.e., the sign is kept
    while the transform is applied to the value.

    Parameters
    ----------
    bounds : 2-tuple of finite float
        Initial bracket (lower, upper) for the optimization range
        when fitting the value of lambda. Default = unbounded.
        Ignored if ``method == "fixed"``.
        For half-open bounds pass a large bound value, e.g., (0, 1e12) for positive
        lambda. Infinity and nan as bound values are not supported.
    method : {"pearsonr", "mle", "guerrero", "fixed"}, default="mle"
        The optimization approach used to determine the lambda value used
        in the Box-Cox transformation.
    sp : int, optional, must be provided (only) if method="guerrero"
        Seasonal periodicity of the data in integer form. Only used if
        method="guerrero" is chosen. Must be an integer >= 2.
    lambda_fixed : float, optional, default = 0.0
        must be provided (only) if method="fixed"
        default means that BoxCoxTransformer behaves like logarithm
    enforce_positive : bool, optional, default = True
        If ``True`, in ``fit`` negative entries of ``X``
        are replaced by their absolute values. In ``transform``, the transform
        is applied to the absolute value while the sign is kept.
        If ``False``, any negative values will be passed unchanged to the
        underlying functions (possibly causing error).

    Attributes
    ----------
    lambda_ : float
        The Box-Cox lambda parameter that was fitted, based on the supplied
        ``method`` and data provided in ``fit``.

    See Also
    --------
    LogTransformer :
        Transformer input data using natural log. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.
    sktime.transformations.series.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    References
    ----------
    .. [1] Box, G. E. P. & Cox, D. R. (1964) An analysis of transformations,
       Journal of the Royal Statistical Society, Series B, 26, 211-252.
    .. [2] V.M. Guerrero, "Time-series analysis supported by Power
       Transformations ", Journal of Forecasting, vol. 12, pp. 37-48, 1993.

    Examples
    --------
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = BoxCoxTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["mloning", "aiwalter", "fkiraly"],
        "python_dependencies": "scipy",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "transform-returns-same-time-index": True,
        "fit_is_empty": False,
        "univariate-only": True,
        "capability:inverse_transform": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        bounds=None,
        method="mle",
        sp=None,
        lambda_fixed=0.0,
        enforce_positive=True,
    ):
        self.bounds = bounds
        self.method = method
        self.sp = sp
        self.lambda_fixed = lambda_fixed
        self.enforce_positive = enforce_positive
        super().__init__()

        VALID_METHODS = ["pearsonr", "mle", "all", "guerrero", "fixed"]

        if method not in VALID_METHODS:
            raise ValueError(
                f"BoxCoxTransformer method must be one of the strings {VALID_METHODS},"
                f" but found {method}"
            )

        if method == "fixed":
            self.set_tags(**{"fit_is_empty": True})
            self.lambda_ = lambda_fixed
        else:
            self.lambda_ = None

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        bounds = self.bounds
        method = self.method
        sp = self.sp
        enforce_positive = self.enforce_positive

        X = X.flatten()

        if enforce_positive:
            X = np.abs(X)

        if self.method in ["pearsonr", "mle", "all"]:
            self.lambda_ = _box_norm(X, bounds, method)
        elif method == "guerrero":
            self.lambda_ = _guerrero(X, sp, bounds)
        elif method == "fixed":
            self.lambda_ = self.lambda_fixed
        else:
            raise RuntimeError(
                f"unreachable state, unexpected method attribute: {method}"
                " this is likely due to method attribute being changed after init"
            )

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        from scipy.special import boxcox

        enforce_positive = self.enforce_positive
        lambda_ = self.lambda_

        X_shape = X.shape
        X = X.flatten()

        if enforce_positive:
            X_sign = np.sign(X)

        Xt = boxcox(np.abs(X), lambda_)

        if enforce_positive:
            Xt = Xt * X_sign

        Xt = Xt.reshape(X_shape)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        from scipy.special import inv_boxcox

        X_shape = X.shape
        Xt = inv_boxcox(X.flatten(), self.lambda_)
        Xt = Xt.reshape(X_shape)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"method": "mle"}
        params2 = {"method": "pearsonr"}
        params3 = {"method": "guerrero", "sp": 2}
        params4 = {"method": "fixed", "lambda_fixed": 1}
        params5 = {"method": "fixed", "lambda_fixed": 0}
        params6 = {"method": "fixed", "lambda_fixed": -1}

        return [params1, params2, params3, params4, params5, params6]


class LogTransformer(BaseTransformer):
    """Natural logarithm transformation.

    The Natural logarithm transformation can be used to make the data more normally
    distributed and stabilize its variance.

    Transforms each data point x to log(scale *(x+offset))

    Parameters
    ----------
    offset : float , default = 0
             Additive constant applied to all the data.
    scale  : float , default = 1
             Multiplicative scaling constant applied to all the data.

    See Also
    --------
    BoxCoxTransformer :
        Applies Box-Cox power transformation. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.
    sktime.transformations.series.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    Notes
    -----
    The log transformation is applied as :math:`ln(y)`.

    Examples
    --------
    >>> from sktime.transformations.series.boxcox import LogTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = LogTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "transform-returns-same-time-index": True,
        "fit_is_empty": True,
        "univariate-only": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, offset=0, scale=1):
        self.offset = offset
        self.scale = scale
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        offset = self.offset
        scale = self.scale
        Xt = np.log(scale * (X + offset))
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        offset = self.offset
        scale = self.scale
        Xt = (np.exp(X) / scale) - offset
        return Xt


def _make_boxcox_optimizer(bounds=None, brack=(-2.0, 2.0)):
    from scipy import optimize

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


# needed for scipy < 1.7.0
# TODO remove in a case where the lower bound is 1.7.0 or higher
def _boxcox_normmax(x, bounds=None, brack=(-2.0, 2.0), method="pearsonr"):
    optimizer = _make_boxcox_optimizer(bounds, brack)

    def _pearsonr(x):
        from scipy import stats
        from scipy.stats import distributions

        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            y = _boxcox(samps, lmbda)
            yvals = np.sort(y)
            r, prob = stats.pearsonr(xvals, yvals)
            return 1 - r

        return optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):
        from scipy.stats import boxcox_llf

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
    """Estimate lambda using the Guerrero method as described in [1]_.

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
    .. [1] V.M. Guerrero, "Time-series analysis supported by Power
       Transformations ", Journal of Forecasting, vol. 12, pp. 37-48, 1993.
    """
    from scipy.stats import variation

    if sp is None or not is_int(sp) or sp < 2:
        raise ValueError(
            "In BoxCoxTransformer, method='guerrero' requires an integer seasonal "
            f"periodicity (sp) value >= 2, but found sp={sp}"
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


def _boxcox(x, lmbda=None, bounds=None):
    r"""Return a dataset transformed by a Box-Cox power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Must be positive 1-dimensional.  Must not be constant.
    lmbda : {None, scalar}, optional
        If ``lmbda`` is not None, do the transformation for that value.
        If ``lmbda`` is None, find the lambda that maximizes the log-likelihood
        function and return it as the second output argument.

    Returns
    -------
    boxcox : ndarray
        Box-Cox power transformed array.
    maxlog : float, optional
        If the ``lmbda`` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.

    See Also
    --------
    probplot, boxcox_normplot, boxcox_normmax, boxcox_llf

    Notes
    -----
    The Box-Cox transform is given by::
        y = (x**lmbda - 1) / lmbda,  for lmbda > 0
            log(x),                  for lmbda = 0
    ``boxcox`` requires the input data to be positive.  Sometimes a Box-Cox
    transformation provides a shift parameter to achieve this; ``boxcox`` does
    not.  Such a shift parameter is equivalent to adding a positive constant to
    ``x`` before calling ``boxcox``.
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
        from scipy.special import boxcox

        return boxcox(x, lmbda)

    # If lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = _box_norm(x, method="mle", bounds=bounds)
    y = _boxcox(x, lmax)

    return y, lmax
