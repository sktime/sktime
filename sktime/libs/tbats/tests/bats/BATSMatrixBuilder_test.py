import pytest
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats import ModelParams, Components, MatrixBuilder

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSMatrixBuilder(object):

    @pytest.mark.parametrize(
        "components, params, expected_w, expected_g",
        [
            [
                dict(),
                dict(alpha=0.5),
                [1],  # expected_w
                [0.5],  # expected g
            ],
            [
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.7, beta=0.6, phi=0.5),
                [1, 0.5],
                [0.7, 0.6],
            ],
            [
                dict(use_trend=True, use_damped_trend=True, seasonal_periods=[3], use_arma_errors=True, p=1, q=1),
                dict(alpha=0.7, beta=0.4, phi=0.7, gamma_params=[0.9], ar_coefs=[0.2], ma_coefs=[0.3]),
                [1, 0.7, 0, 0, 1, 0.2, 0.3],
                [0.7, 0.4, 0.9, 0, 0, 1, 1],
            ],
            [
                dict(seasonal_periods=[4], use_arma_errors=True, p=2),
                dict(alpha=0.7, ar_coefs=[0.3, 0.2], gamma_params=[0.8]),
                [1, 0, 0, 0, 1, 0.3, 0.2],
                [0.7, 0.8, 0, 0, 0, 1, 0],
            ],
            [
                dict(use_trend=True, use_damped_trend=True, seasonal_periods=[3, 5], use_arma_errors=True, p=2, q=3),
                dict(alpha=0.8, beta=0.6, phi=0.7, gamma_params=(0.33, 0.55),
                     ar_coefs=[0.2, 0.1], ma_coefs=[0.34, 0.24, 0.14]),
                [1, 0.7, 0, 0, 1, 0, 0, 0, 0, 1, 0.2, 0.1, 0.34, 0.24, 0.14],
                [0.8, 0.6, 0.33, 0, 0, 0.55, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            ],
        ],
    )
    def test_make_vector(self, components, params, expected_w, expected_g):
        m = MatrixBuilder(
            ModelParams(
                Components(**components),
                **params
            )
        )
        assert np.array_equal(
            expected_w,
            m.make_w_vector()
        )
        assert np.array_equal(
            expected_g,
            m.make_g_vector()
        )

    @pytest.mark.parametrize(
        "components, params, expected_matrix",
        [
            [
                dict(seasonal_periods=[2]),
                dict(alpha=0.5, gamma_params=[0.8]),
                [
                    [0, 1],
                    [1, 0],
                ],
            ],
            [
                dict(seasonal_periods=[3]),
                dict(alpha=0.5, gamma_params=[0.2]),
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
            ],
            [
                dict(seasonal_periods=[2, 3]),
                dict(alpha=0.5, gamma_params=[0.1, 0.2]),
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ],
            ],

        ],
    )
    def test_make_A_matrix(self, components, params, expected_matrix):
        m = MatrixBuilder(ModelParams(Components(**components), **params))
        assert np.array_equal(
            expected_matrix,
            m.make_A_matrix()
        )

    @pytest.mark.parametrize(
        "components, params, expected_matrix",
        [
            [
                dict(),
                dict(alpha=0.5),
                [
                    [1],
                ],
            ],
            [
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=0.7, phi=0.1),
                [
                    [1, 0.1],
                    [0, 0.1],
                ],
            ],
            [
                dict(seasonal_periods=[2, 3], use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=0.7, phi=0.1, gamma_params=[0.2, 0.4]),
                [
                    [1, 0.1, 0, 0, 0, 0, 0],
                    [0, 0.1, 0, 0, 0, 0, 0],
                    [0, 0.0, 0, 1, 0, 0, 0],
                    [0, 0.0, 1, 0, 0, 0, 0],
                    [0, 0.0, 0, 0, 0, 0, 1],
                    [0, 0.0, 0, 0, 1, 0, 0],
                    [0, 0.0, 0, 0, 0, 1, 0],
                ],
            ],
            [
                dict(use_trend=True, use_arma_errors=True, p=2),
                dict(alpha=1, beta=0.5, ar_coefs=[0.4, 0.2]),
                [
                    [1, 1, 0.4, 0.2],
                    [0, 1, 0.2, 0.1],
                    [0, 0, 0.4, 0.2],
                    [0, 0, 1.0, 0.0],
                ],
            ],
            [
                dict(use_trend=True, use_damped_trend=True, seasonal_periods=[2], use_arma_errors=True, p=2, q=1),
                dict(alpha=0.5, beta=1, phi=0.3, gamma_params=[0.25], ar_coefs=[0.4, 0.8], ma_coefs=[1.2]),
                [
                    [1, 0.3, 0, 0, 0.2, 0.4, 0.6],
                    [0, 0.3, 0, 0, 0.4, 0.8, 1.2],
                    [0, 0.0, 0, 1, 0.1, 0.2, 0.3],
                    [0, 0.0, 1, 0, 0.0, 0.0, 0.0],
                    [0, 0.0, 0, 0, 0.4, 0.8, 1.2],
                    [0, 0.0, 0, 0, 1.0, 0.0, 0.0],
                    [0, 0.0, 0, 0, 0.0, 0.0, 0.0],
                ],
            ],
        ],
    )
    def test_make_F_matrix(self, components, params, expected_matrix):
        m = MatrixBuilder(ModelParams(Components(**components), **params))
        assert np.array_equal(
            expected_matrix,
            m.make_F_matrix()
        )
