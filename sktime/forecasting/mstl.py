# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements MSTL."""

__all__ = ["MSTL"]
__authors__ = ["luca-miniati"]

from typing import Optional

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class MSTL(_StatsModelsAdapter):
    """Season-Trend decomposition using LOESS for multiple seasonalities.

    Direct interface for `statsmodels.tsa.seasonal.MSTL`.

    Parameters
    ----------
    endog : array_like
        Data to be decomposed. Must be squeezable to 1-d.
    periods : {int, array_like, None}, optional
        Periodicity of the seasonal components. If None and endog is a pandas Series or
        DataFrame, attempts to determine from endog. If endog is a ndarray, periods
        must be provided.
    windows : {int, array_like, None}, optional
        Length of the seasonal smoothers for each corresponding period. Must be an odd
        integer, and should normally be >= 7 (default). If None then default values
        determined using 7 + 4 * np.arange(1, n + 1, 1) where n is number of seasonal
        components.
    lmbda : {float, str, None}, optional
        The lambda parameter for the Box-Cox transform to be applied to endog prior to
        decomposition. If None, no transform is applied. If “auto”, a value will be
        estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs : dict, optional
        Arguments to pass to STL.

    References
    ----------
    [1] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html

    Examples
    --------
    >>> from sktime.forecasting.mstl import MSTL  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> y = load_airline()
    >>> mstl = MSTL()  # doctest: +SKIP
    >>> res = mstl.fit(y)  # doctest: +SKIP
    >>> res.plot()  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        periods=None,
        windows=None,
        lmbda=None,
        iterate=Optional[int],
        stl_kwargs=Optional[dict],
    ):
        super().__init__()

        self.periods = periods
        self.windows = windows
        self.lmbda = lmbda
        self.iterate = iterate
        self.stl_kwargs = stl_kwargs

    def _fit_forecaster(self, y, X=None):
        from statsmodels.tsa.seasonal import MSTL as _MSTL

        self._forecaster = _MSTL(
            y, self.periods, self.windows, self.lmbda, self.iterate, self.stl_kwargs
        )

        self._fitted_forecaster = self._forecaster.fit()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str , default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict , default = {}
            arameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params
        """
        params1 = {}
        params2 = {
            "periods": [1, 12],
            "windows": 9,
            "lmbda": "auto",
            "iterate": 10,
            "stl_kwargs": {"trend_deg": 0},
        }

        return [params1, params2]
