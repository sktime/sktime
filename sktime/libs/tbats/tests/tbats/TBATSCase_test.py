import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import Components, Case, Context

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSCase(object):

    def create_case(self, **components):
        return Case(Components(**components), Context())

    def test_fit_only_alpha(self):
        alpha = 0.7
        np.random.seed(333)
        T = 200

        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        bats_case = self.create_case(use_trend=False)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)

    def test_fit_with_trend(self):
        alpha = 0.8
        beta = 0.4
        np.random.seed(123)
        T = 400

        b = 0
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + b + d
            l = l + b + alpha * d
            b = b + beta * d

        bats_case = self.create_case(use_trend=True)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.isclose(beta, model.params.beta, atol=0.2)

    def test_fit_trend_and_trigonometric_series(self):
        period_length = 6.5
        T = 300

        l = 0.2
        b = 1.
        s = 1.
        s_star = 0.
        y = [0] * T
        lam = 2 * np.pi / period_length
        for t in range(0, T):
            y[t] = l + s  # no noise
            l = l + b
            s_prev = s
            s = s_prev * np.cos(lam) + s_star * np.sin(lam)
            s_star = - s_prev * np.sin(lam) + s_star * np.cos(lam)

        case = self.create_case(use_trend=True, use_damped_trend=False,
                                seasonal_periods=[period_length], seasonal_harmonics=[1])
        fitted_model = case.fit(y)

        assert np.allclose(y, fitted_model.y_hat, atol=0.01)
