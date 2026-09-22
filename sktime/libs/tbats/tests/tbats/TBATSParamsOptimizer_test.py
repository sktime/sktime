import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import Components, ModelParams, ParamsOptimizer, Context

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSParamsOptimizer(object):

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
                dict(seasonal_periods=[30.02, 365.25], seasonal_harmonics=[1, 2]),
                dict(alpha=0.5, gamma_params=[0.01, 0.02]),
                [-30.9279033, 90.57583425, 19.23927894, -30.08598101, -27.56194995, -0.51760285, -0.94863902],
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

    def test_optimization_with_seasonal_period(self):
        alpha = 0.7
        gamma1 = 0.1
        gamma2 = 0.05
        period_length = 4
        np.random.seed(2342)
        T = 200

        l = 0.2
        s = 1
        s_star = 0
        y = [0] * T
        lam = 2 * np.pi / period_length
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + s + d
            l = l + alpha * d
            s_prev = s
            s = s_prev * np.cos(lam) + s_star * np.sin(lam) + gamma1 * d
            s_star = - s_prev * np.sin(lam) + s_star * np.cos(lam) + gamma2 * d

        c = Components(use_arma_errors=False, seasonal_periods=[period_length], seasonal_harmonics=[1])
        initial_params = ModelParams(components=c, alpha=0.09, gamma_params=[0., 0.])

        optimizer = ParamsOptimizer(self.context)
        optimizer.optimize(y, initial_params)

        assert optimizer.converged()

        model = optimizer.optimal_model()

        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.allclose([gamma1, gamma2], model.params.gamma_params, atol=0.1)

    def test_optimization_with_2_seasonal_periods(self):
        alpha = 0.4
        gamma = 0.05
        period1_length = 4
        period2_length = 12
        np.random.seed(2342)
        T = 200

        l = 0.2
        s1 = 1
        s1_star = 0
        s2 = 0
        s2_star = 1
        y = [0] * T
        lam1 = 2 * np.pi / period1_length
        lam2 = 2 * np.pi / period2_length
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + s1 + s2 + d
            l = l + alpha * d

            s1_prev = s1
            s1 = s1_prev * np.cos(lam1) + s1_star * np.sin(lam1) + gamma * d
            s1_star = - s1_prev * np.sin(lam1) + s1_star * np.cos(lam1)

            s2_prev = s2
            s2 = s2_prev * np.cos(lam2) + s2_star * np.sin(lam2)
            s2_star = - s2_prev * np.sin(lam2) + s2_star * np.cos(lam2)

        c = Components(use_arma_errors=False,
                       seasonal_periods=[period1_length, period2_length], seasonal_harmonics=[1, 1])
        initial_params = ModelParams(components=c, alpha=0.09, gamma_params=[0., 0., 0., 0.])

        optimizer = ParamsOptimizer(self.context)
        optimizer.optimize(y, initial_params)

        assert optimizer.converged()

        model = optimizer.optimal_model()

        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.allclose([gamma, 0., 0., 0.], model.params.gamma_params, atol=0.25)
