import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats import ModelParams, Components

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSModelParams(object):

    @pytest.mark.parametrize(
        "components, params, expected",
        [
            [
                dict(),
                dict(alpha=0.7),
                [0.7],
            ],
            [
                dict(use_trend=True, use_box_cox=True),
                dict(alpha=0.7, beta=0.5, box_cox_lambda=0.3),
                [0.7, 0.3, 0.5],
            ],
            [
                dict(use_trend=True, use_damped_trend=True, use_arma_errors=True, p=2),
                dict(alpha=0.7, beta=0.5, phi=0.2, ar_coefs=[-0.1, -0.2]),
                [0.7, 0.5, 0.2, -0.1, -0.2],
            ],
            [
                dict(use_trend=True, use_damped_trend=True, seasonal_periods=[7, 12], use_arma_errors=True, p=1, q=1),
                dict(alpha=0.7, beta=0.5, phi=0.2, gamma_params=[2, 1], ar_coefs=[-0.3], ma_coefs=[-0.1]),
                [0.7, 0.5, 0.2, 2, 1, -0.3, -0.1],
            ],
        ]
    )
    def test_to_vector(self, components, params, expected):
        c = Components(**components)
        p = ModelParams(c, **params)
        vector = p.to_vector()
        assert np.array_equal(expected, vector)

    def test_with_vector_values(self):
        c = Components(use_trend=True, use_damped_trend=True,
                       use_arma_errors=True, use_box_cox=True, seasonal_periods=[7, 30], p=1, q=2)
        p = ModelParams(c, alpha=0.1, box_cox_lambda=1.4, beta=0.2, phi=0.3, gamma_params=[0.4, 0.5],
                        ar_coefs=[-0.3], ma_coefs=[-0.4, 0.6])
        v = p.to_vector()
        assert np.array_equal([0.1, 1.4, 0.2, 0.3, 0.4, 0.5, -0.3, -0.4, 0.6], v)

        p = ModelParams(c, alpha=0.9, box_cox_lambda=1.4)
        p = p.with_vector_values(v)
        assert p.alpha == 0.1
        assert p.box_cox_lambda == 1.4
        assert p.beta == 0.2
        assert p.phi == 0.3
        assert np.array_equal([0.4, 0.5], p.gamma_params)
        assert np.array_equal([-0.3], p.ar_coefs)
        assert np.array_equal([-0.4, 0.6], p.ma_coefs)

    @pytest.mark.parametrize(
        "components, params, expected",
        [
            [
                dict(),
                dict(alpha=1),
                [0.],
            ],
            [
                dict(use_trend=True),
                dict(alpha=1, beta=0.0),
                [0., 0.],
            ],
            [
                dict(use_trend=True, use_arma_errors=True, p=2),
                dict(alpha=1, beta=0.1, ar_coefs=[0.1, 0.2]),
                [0.] * 4,
            ],
            [
                dict(use_trend=True, seasonal_periods=[3, 4], use_arma_errors=True, q=1),
                dict(alpha=1, beta=0.1, ma_coefs=[0.3]),
                [0.] * 10,
            ],
        ]
    )
    def test_create_x0_of_zeroes(self, components, params, expected):
        c = Components(**components)
        p = ModelParams(c, **params)
        x0 = p._create_x0_of_zeroes()
        assert np.array_equal(expected, x0)

    @pytest.mark.parametrize(
        "components, params, expected_gamma",
        [
            [  # no seasonal periods, empty gammas
                dict(),
                dict(alpha=1),
                [],
            ],
            [  # one seasonal period
                dict(seasonal_periods=[4.2]),
                dict(alpha=1, gamma_params=[0.5]),
                [0.5],
            ],
            [  # 3 periods, 3 coefs in each gamma vector
                dict(seasonal_periods=[3, 4, 5]),
                dict(alpha=1, gamma_params=[1.3, 2.4, 3.5]),
                [1.3, 2.4, 3.5],
            ],
            [  # should initialize gamma to zeros
                dict(seasonal_periods=[3, 4, 5]),
                dict(alpha=1, beta=0.2),
                [0.] * 3,
            ],
        ]
    )
    def test_gamma(self, components, params, expected_gamma):
        c = Components(**components)
        p = ModelParams(c, **params)

        assert np.array_equal(expected_gamma, p.gamma_params)
