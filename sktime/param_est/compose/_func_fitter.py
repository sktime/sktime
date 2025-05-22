"""Implements FunctionParamFitter, a class to create custom parameter fitters."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.param_est.base import BaseParamFitter

__author__ = ["tpvasconcelos"]
__all__ = ["FunctionParamFitter"]


class FunctionParamFitter(BaseParamFitter):
    r"""Constructs a parameter fitter from an arbitrary callable.

    A FunctionParamFitter forwards its X argument to a user-defined
    function (or callable object) and sets the result of this function
    to the ``param`` attribute. This can be useful for stateless
    estimators such as simple conditional parameter selectors.

    Note: If a lambda function is used as the ``func``, then the
    resulting estimator will not be pickleable.

    Parameters
    ----------
    param : str
        The name of the parameter to set.
    func : callable (X: X_type, **kwargs) -> Any
        The callable to use for the parameter estimation. This will be
        passed the same arguments as estimator, with args and kwargs
        forwarded.
    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.
    X_type : str, one of "pd.DataFrame, pd.Series, np.ndarray", or list thereof
        default = ["pd.DataFrame", "pd.Series", "np.ndarray"]
        list of types that func is assumed to allow for X (see signature above)
        if X passed to transform/inverse_transform is not on the list,
            it will be converted to the first list element before passed to funcs

    See Also
    --------
    sktime.param_est.plugin.PluginParamsForecaster :
        Plugs parameters from a parameter estimator into a forecaster.
    sktime.forecasting.compose.MultiplexForecaster :
        MultiplexForecaster for selecting among different models.

    Examples
    --------
    This class could be used to construct a parameter estimator that
    selects a forecaster based on the input data's length. The
    selected forecaster can be stored in the ``selected_forecaster_``
    attribute, which can be then passed down to a
    :class:`~sktime.forecasting.compose.MultiplexForecaster` via a
    :class:`~sktime.param_est.plugin.PluginParamsForecaster`.

    >>> import numpy as np
    >>> from sktime.param_est.compose import FunctionParamFitter
    >>> param_est = FunctionParamFitter(
    ...     param="selected_forecaster",
    ...     func=(
    ...         lambda X, threshold: "naive-seasonal"
    ...         if len(X) >= threshold
    ...         else "naive-last"
    ...     ),
    ...     kw_args={"threshold": 7},
    ... )
    >>> param_est.fit(np.asarray([1, 2, 3, 4]))
    FunctionParamFitter(...)
    >>> param_est.get_fitted_params()
    {'selected_forecaster': 'naive-last'}
    >>> param_est.fit(np.asarray([1, 2, 3, 4, 5, 6, 7]))
    FunctionParamFitter(...)
    >>> param_est.get_fitted_params()
    {'selected_forecaster': 'naive-seasonal'}

    The full conditional forecaster selection pipeline could look
    like this:

    >>> from sktime.forecasting.compose import MultiplexForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.param_est.plugin import PluginParamsForecaster
    >>> forecaster = PluginParamsForecaster(
    ...     param_est=param_est,
    ...     forecaster=MultiplexForecaster(
    ...         forecasters=[
    ...             ("naive-last", NaiveForecaster()),
    ...             ("naive-seasonal", NaiveForecaster(sp=7)),
    ...         ]
    ...     ),
    ... )
    >>> forecaster.fit(np.asarray([1, 2, 3, 4]))
    PluginParamsForecaster(...)
    >>> forecaster.predict(fh=[1,2,3])
    array([[4.],
           [4.],
           [4.]])
    >>> forecaster.fit(np.asarray([1, 2, 3, 4, 5, 6, 7]))
    PluginParamsForecaster(...)
    >>> forecaster.predict(fh=[1,2,3])
    array([[1.],
           [2.],
           [3.]])
    """

    _tags = {
        "authors": ["tpvasconcelos"],
        "maintainers": ["tpvasconcelos"],
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "scitype:X": ["Series", "Panel", "Hierarchical"],
        "capability:missing_values": True,
        "capability:multivariate": False,
    }

    def __init__(self, param, func, kw_args=None, X_type=None):
        self.param = param
        self.func = func
        self.kw_args = kw_args
        self.X_type = X_type
        super().__init__()

        if X_type is not None:
            self.set_tags(X_inner_mtype=X_type)

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        param = self.param.rstrip("_") + "_"
        setattr(self, param, self.func(X, **(self.kw_args or {})))
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are no reserved values for parameter estimators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"param": "param", "func": _lambda_test_simple},
            {"param": "param", "func": _lambda_test_kwarg, "kw_args": {"kwarg": 1}},
        ]
        return params


def _lambda_test_simple(X):
    return "foo"


def _lambda_test_kwarg(X, kwarg):
    return "foo"
