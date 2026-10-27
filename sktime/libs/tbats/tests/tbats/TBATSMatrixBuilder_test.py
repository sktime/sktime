import pytest
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import ModelParams, Components, MatrixBuilder

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSMatrixBuilder(object):

    @pytest.mark.parametrize(
        "components, params, expected_w, expected_g",
        [
            [
                dict(),
                dict(alpha=0.5),
                [1],  # expected_w
                [0.5],  # expected g
            ],
            [  # no seasonal harmonics, no seasonal parameters
                dict(seasonal_periods=[2, 3], seasonal_harmonics=[0, 0]),
                dict(alpha=0.5),
                [1],  # expected_w
                [0.5],  # expected g
            ],
            [
                dict(seasonal_periods=[3], seasonal_harmonics=[2], use_arma_errors=True, p=1, q=1),
                dict(alpha=0.7, gamma_params=[0.9, 0.8], ar_coefs=[0.2], ma_coefs=[0.3]),
                [1, 1, 1, 0, 0, 0.2, 0.3],
                [0.7, 0.9, 0.9, 0.8, 0.8, 1, 1],
            ],
            [
                dict(use_trend=True, seasonal_periods=[3, 5], seasonal_harmonics=[1, 4],
                     use_arma_errors=True, p=2, q=3),
                dict(alpha=0.8, beta=0.6, phi=0.7, gamma_params=[0.3, 0.5, 0.4, 0.6],
                     ar_coefs=[0.2, 0.1], ma_coefs=[0.34, 0.24, 0.14]),
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0.2, 0.1, 0.34, 0.24, 0.14],
                [0.8, 0.6, 0.3, 0.5, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 1, 0, 1, 0, 0],
            ],
            [  # when harmonics is 0 it does not produce parameters
                dict(use_trend=True, seasonal_periods=[3, 5], seasonal_harmonics=[1, 0],
                     use_arma_errors=True, p=2, q=3),
                dict(alpha=0.8, beta=0.6, phi=0.7, gamma_params=[0.3, 0.5, 0.4, 0.6],
                     ar_coefs=[0.2, 0.1], ma_coefs=[0.34, 0.24, 0.14]),
                [1, 1, 1, 0, 0.2, 0.1, 0.34, 0.24, 0.14],
                [0.8, 0.6, 0.3, 0.5, 1, 0, 1, 0, 0],
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
        "components, params, expected_gamma",
        [
            [  # no periods means no gamma
                dict(), dict(alpha=0.5), [],
            ],
            [
                dict(
                    seasonal_periods=[7],
                    seasonal_harmonics=[1],
                ),
                dict(
                    alpha=0.5,
                    gamma_params=[1.1, 1.2],
                ),
                [1.1, 1.2],
            ],
            [  # each gamma is repeated according to amount of harmonics
                dict(
                    seasonal_periods=[3, 5],
                    seasonal_harmonics=[2, 3],
                ),
                dict(
                    alpha=0.5,
                    gamma_params=[1.1, 1.2, 2.1, 2.2]
                ),
                [
                    1.1, 1.1,
                    1.2, 1.2,
                    2.1, 2.1, 2.1,
                    2.2, 2.2, 2.2,
                ],
            ],
            [  # when there are no harmonics for a season, then respective gamma is not present
                dict(
                    seasonal_periods=[3, 5],
                    seasonal_harmonics=[0, 3],
                ),
                dict(
                    alpha=0.5,
                    gamma_params=[1.1, 1.2, 2.1, 2.2],
                ),
                [
                    2.1, 2.1, 2.1,
                    2.2, 2.2, 2.2,
                ],
            ],
        ]
    )
    def test_make_gamma_vector(self, components, params, expected_gamma):
        m = MatrixBuilder(ModelParams(components=Components(**components), **params))
        assert np.array_equal(expected_gamma, m.make_gamma_vector())

    @pytest.mark.parametrize(
        "components, params, expected_matrix",
        [
            [
                dict(seasonal_periods=[7], seasonal_harmonics=[1]),
                dict(alpha=0.5, gamma_params=[0.8, 0.9]),
                [
                    [np.cos(2 * np.pi / 7), np.sin(2 * np.pi / 7)],
                    [-np.sin(2 * np.pi / 7), np.cos(2 * np.pi / 7)],
                ],
            ],
            [
                dict(seasonal_periods=[365], seasonal_harmonics=[3]),
                dict(alpha=0.5, gamma_params=[0.2, 0.3]),
                [
                    [np.cos(1 * 2 * np.pi / 365), 0, 0, np.sin(1 * 2 * np.pi / 365), 0, 0],
                    [0, np.cos(2 * 2 * np.pi / 365), 0, 0, np.sin(2 * 2 * np.pi / 365), 0],
                    [0, 0, np.cos(3 * 2 * np.pi / 365), 0, 0, np.sin(3 * 2 * np.pi / 365)],
                    [-np.sin(1 * 2 * np.pi / 365), 0, 0, np.cos(1 * 2 * np.pi / 365), 0, 0],
                    [0, -np.sin(2 * 2 * np.pi / 365), 0, 0, np.cos(2 * 2 * np.pi / 365), 0],
                    [0, 0, -np.sin(3 * 2 * np.pi / 365), 0, 0, np.cos(3 * 2 * np.pi / 365)],
                ],
            ],
            [
                dict(seasonal_periods=[2, 4], seasonal_harmonics=[1, 1]),
                dict(alpha=0.5, gamma_params=[0.1, 0.2]),
                [
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, -1, 0],
                ],
            ],

        ],
    )
    def test_make_A_matrix(self, components, params, expected_matrix):
        m = MatrixBuilder(ModelParams(Components(**components), **params))
        assert np.allclose(
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
                dict(seasonal_periods=[100, 200], seasonal_harmonics=[0, 0]),
                dict(alpha=0.5),
                [
                    [1],
                ],
            ],
            [
                dict(seasonal_periods=[2, 3], seasonal_harmonics=[2, 1], use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=0.7, phi=0.1, gamma_params=[0.2, 0.4, 0.3, 0.5]),
                [
                    [1, 0.1, 0, 0, 0, 0, 0, 0],
                    [0, 0.1, 0, 0, 0, 0, 0, 0],
                    [0, 0.0, np.cos(2 * np.pi / 2), 0, np.sin(2 * np.pi / 2), 0, 0, 0],
                    [0, 0.0, 0, np.cos(4 * np.pi / 2), 0, np.sin(4 * np.pi / 2), 0, 0],
                    [0, 0.0, -np.sin(2 * np.pi / 2), 0, np.cos(2 * np.pi / 2), 0, 0, 0],
                    [0, 0.0, 0, -np.sin(4 * np.pi / 2), 0, np.cos(4 * np.pi / 2), 0, 0],
                    [0, 0.0, 0, 0, 0, 0, np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
                    [0, 0.0, 0, 0, 0, 0, -np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)],
                ],
            ],
            [
                dict(use_trend=True, use_damped_trend=True,
                     seasonal_periods=[4], seasonal_harmonics=[1],
                     use_arma_errors=True, p=2, q=1),
                dict(alpha=0.5, beta=1, phi=0.3, gamma_params=[0.25, 0.1], ar_coefs=[0.4, 0.8], ma_coefs=[1.2]),
                [
                    [1, 0.3, 0, 0, 0.2, 0.4, 0.6],
                    [0, 0.3, 0, 0, 0.4, 0.8, 1.2],
                    [0, 0.0, np.cos(2 * np.pi / 4), np.sin(2 * np.pi / 4), 0.25 * 0.4, 0.25 * 0.8, 0.25 * 1.2],
                    [0, 0.0, -np.sin(2 * np.pi / 4), np.cos(2 * np.pi / 4), 0.1 * 0.4, 0.1 * 0.8, 0.1 * 1.2],
                    [0, 0.0, 0, 0, 0.4, 0.8, 1.2],
                    [0, 0.0, 0, 0, 1.0, 0.0, 0.0],
                    [0, 0.0, 0, 0, 0.0, 0.0, 0.0],
                ],
            ],
        ],
    )
    def test_make_F_matrix(self, components, params, expected_matrix):
        m = MatrixBuilder(ModelParams(Components(**components), **params))
        F = m.make_F_matrix()
        assert np.allclose(
            expected_matrix,
            F
        )
