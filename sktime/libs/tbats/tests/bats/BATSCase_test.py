import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed

from sktime.libs.tbats.bats import Components, Case, Context

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSCase(object):

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

    def test_fit_bats_case_with_trend(self):
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

    def test_fit_bats_case_with_damped_trend(self):
        alpha = 0.4
        beta = 0.6
        phi = 0.9
        np.random.seed(987)
        T = 500

        b = 0
        b_long = 0.0
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal(scale=1.0)
            y[t] = l + b + d
            l = l + b + alpha * d
            b = (1 - phi) * b_long + phi * b + beta * d

        bats_case = self.create_case(use_trend=True, use_damped_trend=True)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.isclose(beta, model.params.beta, atol=0.1)
        assert np.isclose(phi, model.params.phi, atol=0.1)

    def test_fit_with_boxcox(self):
        np.random.seed(49385)
        y = (np.random.uniform(size=100) + np.asarray(
            range(40, 140)) / 20) ** 2  # variance is increasing in those series

        bats_case = self.create_case(use_box_cox=True)
        model = bats_case.fit(y)

        assert np.isclose(0.4948, model.params.box_cox_lambda, atol=0.01)
        assert not np.allclose(model.resid, model.resid_boxcox)
