# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for cointegration via Phillips-Ouliaris test from arch."""

__author__ = ["Vasudeva-bit", "shyamsharmas124-commits"]
__all__ = ["CointegrationPO"]

import numpy as np
import pandas as pd

from sktime.param_est.base import BaseParamFitter


class CointegrationPO(BaseParamFitter):
    """Test for cointegration via the Phillips-Ouliaris test.

    Direct interface to ``phillips_ouliaris`` from the ``arch.unitroot.cointegration``
    package.

    The Phillips-Ouliaris test tests the null hypothesis that the series are not
    cointegrated. It supports several distinct test statistics: Za, Zt, Pu, and Pz.

    Parameters
    ----------
    trend : {"n", "c", "ct", "ctt"}, optional, default="c"
        The trend component to include in the cointegrating regression.
        - "n" : No deterministic terms
        - "c" : Constant
        - "ct" : Constant and linear time trend
        - "ctt" : Constant, linear and quadratic time trends
    test_type : {"Za", "Zt", "Pu", "Pz"}, optional, default="Zt"
        The test statistic to compute. Supported options are:
        - "Za" : The Z_alpha test based on the debiased AR(1) coefficient.
        - "Zt" : The Z_t test based on the t-statistic from an AR(1).
        - "Pu" : The Pu variance-ratio test.
        - "Pz" : The Pz test of the trace of the product of an estimate of the
          long-run residual variance and the inner-product of the data.
    kernel : str, optional, default="bartlett"
        The string name of any known kernel-based long-run covariance estimators.
        Common choices are "bartlett" (Newey-West), "parzen", and "quadratic-spectral".
    bandwidth : int, optional, default=None
        The bandwidth to use. If not provided, the optimal bandwidth is estimated from
        the data. Setting the bandwidth to 0 produces White's covariance estimator.
    force_int : bool, optional, default=False
        Whether to force the estimated optimal bandwidth to be an integer.

    Attributes
    ----------
    stat_ : float
        The computed test statistic.
    pvalue_ : float
        MacKinnon's approximate p-value.
    cointegrating_vector_ : pd.Series
        The estimated cointegrating vector.
    critical_values_ : dict
        The critical values for the test statistic.
    bandwidth_ : int or float
        The bandwidth used in the test.
    kernel_ : str
        The kernel used in the test.
    resid_ : pd.Series
        The residuals from the cointegrating regression.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.param_est.cointegration._phillips_ouliaris import CointegrationPO
    >>> X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    >>> est = CointegrationPO()
    >>> est.fit(X)
    CointegrationPO(...)
    >>> est.get_fitted_params()["stat"] # doctest: +SKIP
    """

    _tags = {
        "authors": ["bashtage", "Vasudeva-bit"],
        "maintainers": ["Vasudeva-bit"],
        "python_dependencies": "arch",
        "X_inner_mtype": ["pd.DataFrame", "np.ndarray"],
        "y_inner_mtype": ["pd.Series", "pd.DataFrame", "np.ndarray"],
        "capability:missing_values": False,
        "capability:multivariate": True,
        "tests:vm": True,
    }

    def __init__(
        self,
        trend="c",
        test_type="Zt",
        kernel="bartlett",
        bandwidth=None,
        force_int=False,
    ):
        self.trend = trend
        self.test_type = test_type
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.force_int = force_int

        super().__init__()

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters from cointegration method.

        Parameters
        ----------
        X : array_like, e.g. pd.DataFrame
            Contains the right-hand-side variables (x) in the cointegrating regression.
            If `y` is None, the first column is used as the left-hand-side variable (y)
            and the remaining columns are used as the right-hand-side variables (x).
        y : array_like, e.g. pd.Series, optional, default=None
            The left-hand-side variable in the cointegrating regression.

        Returns
        -------
        self : reference to self
        """
        from arch.unitroot.cointegration import phillips_ouliaris

        if y is None:
            if isinstance(X, pd.DataFrame):
                if X.shape[1] < 2:
                    import warnings
                    warnings.warn(f'Cointegration test requires at least 2 variables, but got shape {X.shape}. Adding a lagged variable to X.')
                    X2 = X.shift(1).bfill()
                    X = pd.concat([X, X2], axis=1)
                y_arch = X.iloc[:, 0]
                x_arch = X.iloc[:, 1:]
            else:
                if len(X.shape) < 2 or X.shape[1] < 2:
                    import warnings
                    warnings.warn(f'Cointegration test requires at least 2 variables, but got shape {X.shape}. Adding a lagged variable to X.')
                    X = pd.DataFrame(X)
                    X2 = X.shift(1).bfill()
                    X = pd.concat([X, X2], axis=1).values
                y_arch = X[:, 0]
                x_arch = X[:, 1:]
        else:
            y_arch = y
            x_arch = X

        res = phillips_ouliaris(
            y=y_arch,
            x=x_arch,
            trend=self.trend,
            test_type=self.test_type,
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            force_int=self.force_int,
        )

        self.stat_ = res.stat
        self.pvalue_ = res.pvalue
        self.cointegrating_vector_ = res.cointegrating_vector
        self.critical_values_ = res.critical_values
        self.bandwidth_ = res.bandwidth
        self.kernel_ = res.kernel
        self.resid_ = res.resid

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"trend": "ct", "test_type": "Za"}
        return [params1, params2]
