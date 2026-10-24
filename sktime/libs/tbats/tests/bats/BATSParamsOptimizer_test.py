import numpy as np
import pytest
import scipy.stats as stats
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats import Components, ModelParams, ParamsOptimizer, Context

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSParamsOptimizer(object):

    def setup_method(self, method):
        self.context = Context()

    @pytest.mark.parametrize(
        "components, params, expected",
        [
            [
                dict(),
                dict(alpha=0.5),
                [2.57142857],
            ],
            [
                dict(use_trend=True),
                dict(alpha=0.5, beta=0.7, phi=1),
                [2.53968254, 0.51851852],
            ],
            [
                dict(use_trend=True, use_arma_errors=True, p=2, q=1),
                dict(alpha=0.5, beta=0.7, ar_coefs=np.array([0.0, 0.0]), ma_coefs=np.array([0.0])),
                [2.53968254, 0.51851852, 0., 0., 0.],
            ],
            [
                dict(use_trend=True, use_arma_errors=True, p=2, q=1),
                dict(alpha=0.5, beta=0.7, ar_coefs=np.array([0.7, 0.2]), ma_coefs=np.array([0.3])),
                [-0.91428571, 4.62857143, 0., 0., 0.],
            ],
        ]
    )
    def test_calculate_seed_x0(self, components, params, expected):
        y = [2.0, 4.0, 2.0]
        c = Components(**components)
        p = ModelParams(c, **params)
        m = ParamsOptimizer(self.context)
        x0 = m._calculate_seed_x0(y, p)
        assert np.allclose(expected, x0)

    def test_fit_alpha_only(self):
        alpha = 0.7
        np.random.seed(345)
        T = 200

        l = 0.2
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        c = Components(use_arma_errors=False)
        p = ModelParams(c, alpha=0.09)  # default starting value for alpha
        optimizer = ParamsOptimizer(self.context)
        optimizer.optimize(y, p)
        fitted_model = optimizer.optimal_model()
        resid = fitted_model.resid

        # AGAIN why x0 estimation is so shitty, l is far from being 0.2

        # Residuals should form a normal distribution
        _, pvalue = stats.normaltest(resid)
        assert 0.05 < pvalue  # large p-value, we can not reject null hypothesis of normal distribution

        # Mean of residuals should be close to 0
        _, pvalue = stats.ttest_1samp(resid, popmean=0.0)
        assert 0.05 < pvalue  # large p-value we can not reject null hypothesis that mean is 0

        # We expect 95% of residuals to lie within [-2,2] interval
        assert len(resid[np.where(np.abs(resid) < 2)]) / len(resid) > 0.90
