import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.abstract import ComponentMatrix
from sktime.libs.tbats.tbats import Components

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSComponentMatrix(object):

    @pytest.mark.parametrize(
        "components, matrix, expected_break",
        [
            [  # no periods means no break
                dict(), [1], [],
            ],
            [  # one season breaks into one period
                dict(seasonal_periods=[3], seasonal_harmonics=[2]),
                [
                    [1, 10.1, 10.2, 10.3, 10.4],  # alpha, 4 params for season
                    [1, 20.1, 20.2, 20.3, 10.5],
                    [1, 30.1, 30.2, 30.3, 10.6],
                    [1, 40.1, 40.2, 40.3, 10.7],
                ],
                [
                    [
                        [10.1, 10.2, 10.3, 10.4],
                        [20.1, 20.2, 20.3, 10.5],
                        [30.1, 30.2, 30.3, 10.6],
                        [40.1, 40.2, 40.3, 10.7],
                    ],
                ],
            ],
            [  # two seasons break into 2 matrices of proper length
                dict(seasonal_periods=[2, 3], seasonal_harmonics=[1, 1]),
                [[1, 1.1, 1.2, 0.1, 0.2]],
                [
                    [[1.1, 1.2]],
                    [[0.1, 0.2]],
                ],
            ],
        ]
    )
    def test_break_into_seasons(self, components, matrix, expected_break):
        components = Components(**components)
        matrix_obj = ComponentMatrix(matrix, components)
        seasons = matrix_obj.break_into_seasons()
        assert len(expected_break) == len(seasons)
        for i in range(0, len(expected_break)):
            assert np.array_equal(expected_break[i], seasons[i])

    @pytest.mark.parametrize(
        "components, matrix, expected_alpha_beta, expected_seasonal, expected_arma",
        [
            [  # split into alpha_beta, seasonal and arma
                dict(use_trend=True, use_damped_trend=True, use_arma_errors=True, p=1),
                [
                    [1., 2., 3.],  # alpha, beta, p1
                    [4., 5., 6.],
                ],
                [
                    [1., 2.],  # alpha, beta
                    [4., 5.],
                ],
                [
                    [],  # seasonal
                    [],
                ],
                [
                    [3.],  # alpha, beta, p1
                    [6.],
                ],
            ],
            [  # for vector
                dict(seasonal_periods=(2.2, 3.3), seasonal_harmonics=[1, 2], use_arma_errors=True, q=1),
                [
                    [1., 11., 12., 21., 22., 23., 24., 3.],  # alpha, s11, s12, s21, s22, s23, s24, q1
                ],
                [
                    [1.],  # alpha
                ],
                [
                    [11., 12., 21., 22., 23., 24.],  # seasonal
                ],
                [
                    [3.],  # q1
                ],
            ],
        ]
    )
    def test_part_split(self, components, matrix, expected_alpha_beta, expected_seasonal, expected_arma):
        components = Components(**components)
        matrix_obj = ComponentMatrix(matrix, components)
        assert np.array_equal(expected_alpha_beta, matrix_obj.alpha_beta_part())
        assert np.array_equal(expected_seasonal, matrix_obj.seasonal_part())
        assert np.array_equal(expected_arma, matrix_obj.arma_part())
