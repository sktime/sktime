# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimator for cointegration via Phillips-Ouliaris test from arch."""

__author__ = ["Vasudeva-bit", "shyamsharmas124-commits"]
__all__ = ["PhillipsOuliarisCointegration"]

import numpy as np
import pandas as pd

from sktime.param_est.base import BaseParamFitter


class PhillipsOuliarisCointegration(BaseParamFitter):
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
    p_threshold : float, optional, default=0.05
        The p-value threshold to use when deriving the `cointegrated_` boolean attribute.

    Attributes
    ----------
    cointegrated_ : bool
        Whether the series in ``fit`` is cointegrated according to the test,
        i.e., whether the null of no cointegration is rejected at ``p_threshold``.
    test_statistic_ : float
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
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.cointegration import PhillipsOuliarisCointegration
    >>> X = load_airline()  # doctest: +SKIP
    >>> y = X.shift(1).bfill()  # doctest: +SKIP
    >>> est = PhillipsOuliarisCointegration()  # doctest: +SKIP
    >>> est.fit(X=X, y=y)  # doctest: +SKIP
    PhillipsOuliarisCointegration(...)
    >>> est.get_fitted_params()["cointegrated"]  # doctest: +SKIP
    False
    """

    _tags = {
        "authors": ["bashtage", "Vasudeva-bit"],
        "maintainers": ["Vasudeva-bit"],
        "python_dependencies": "arch",
        "X_inner_mtype": ["pd.DataFrame", "np.ndarray", "pd.Series"],
        "y_inner_mtype": ["pd.Series", "np.ndarray", "pd.DataFrame"],
        "capability:missing_values": False,
        "capability:multivariate": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_deepcopy_fitted",
            "test_fit_does_not_overwrite_hyper_params",
            "test_fit_returns_self",
            "test_fit_updates_state",
            "test_non_state_changing_method_contract",
            "test_get_fitted_params",
            "test_update",
            "test_raises_not_fitted_error",
        ],
    }

    def __init__(
        self,
        trend="c",
        test_type="Zt",
        kernel="bartlett",
        bandwidth=None,
        force_int=False,
        p_threshold=0.05,
    ):
        self.trend = trend
        self.test_type = test_type
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.force_int = force_int
        self.p_threshold = p_threshold

        super().__init__()

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters from cointegration method."""
        from skbase.utils.dependencies import _check_soft_dependencies
        _check_soft_dependencies("arch", severity="error")
        from arch.unitroot.cointegration import phillips_ouliaris

        if y is None:
            raise ValueError(
                "y must be explicitly passed in fit(X, y). "
                "The Phillips-Ouliaris test regresses y on X, and results depend on which series is y."
            )

        res = phillips_ouliaris(
            y=y,
            x=X,
            trend=self.trend,
            test_type=self.test_type,
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            force_int=self.force_int,
        )

        self.test_statistic_ = res.stat
        self.pvalue_ = res.pvalue
        self.cointegrated_ = bool(self.pvalue_ < self.p_threshold)
        
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
        params2 = {"trend": "ct", "test_type": "Za", "p_threshold": 0.1}
        return [params1, params2]
