import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.abstract import ComponentMatrix
from sktime.libs.tbats.bats import Components

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestComponentMatrix(object):

    @pytest.mark.parametrize(
        "seasonal_periods, matrix, expected_break",
        [
            [  # no periods means no break
                [], [1], [],
            ],
            [  # should work with 1D vectors
                [2, 3],
                [1, 10.1, 10.2, 0.1, 0.2, 0.3],
                [
                    [
                        [10.1, 10.2],
                    ],
                    [
                        [0.1, 0.2, 0.3],
                    ],
                ],
            ],
            [  # one season breaks into one period
                [3],
                [
                    [1, 10.1, 10.2, 10.3],  # alpha, 3 params for season
                    [1, 20.1, 20.2, 20.3],
                    [1, 30.1, 30.2, 30.3],
                    [1, 40.1, 40.2, 40.3],
                ],
                [
                    [
                        [10.1, 10.2, 10.3],
                        [20.1, 20.2, 20.3],
                        [30.1, 30.2, 30.3],
                        [40.1, 40.2, 40.3],
                    ],
                ],
            ],
            [  # two seasons break into 2 matrices of proper length
                [2, 3],
                [[1, 1.1, 1.2, 0.1, 0.2, 0.3]],
                [
                    [[1.1, 1.2]],
                    [[0.1, 0.2, 0.3]],
                ],
            ],
            [  # for 3 seasons, we expect 3 matrices of proper length
                [2, 3, 5],
                [
                    [1, 11.1, 11.2, 10.1, 10.2, 10.3, 11., 12., 13., 14., 15.],
                    [1, 21.1, 21.2, 20.1, 20.2, 20.3, 21., 22., 23., 24., 25.],
                ],
                [
                    [
                        [11.1, 11.2],
                        [21.1, 21.2],
                    ],
                    [
                        [10.1, 10.2, 10.3],
                        [20.1, 20.2, 20.3],
                    ],
                    [
                        [11., 12., 13., 14., 15.],
                        [21., 22., 23., 24., 25.],
                    ]
                ],
            ],
        ]
    )
    def test_break_into_seasons(self, seasonal_periods, matrix, expected_break):
        components = Components(seasonal_periods=seasonal_periods)
        matrix_obj = ComponentMatrix(matrix, components)
        seasons = matrix_obj.break_into_seasons()
        assert len(expected_break) == len(seasons)
        for i in range(0, len(expected_break)):
            assert np.array_equal(expected_break[i], seasons[i])

    @pytest.mark.parametrize(
        "components, matrix, new_seasonal_part, new_seasonal_periods, expected_matrix",
        [
            [  # inject seasonal into matrix that has had no seasonal yet
                dict(use_trend=True, use_arma_errors=True, p=1),
                [
                    [1., 2., 3.],  # alpha, beta, p1
                    [4., 5., 6.],
                ],
                [
                    [1.1, 1.2],
                    [2.1, 2.2],
                ],
                [2],
                [
                    [1., 2., 1.1, 1.2, 3.],  # alpha, beta, seasonal1, seasonal2, p1
                    [4., 5., 2.1, 2.2, 6.],
                ],
            ],
            [  # remove seasonal
                dict(use_arma_errors=True, seasonal_periods=(2, 3), q=1),
                [
                    [1., 11., 12., 21., 22., 23., 3.],  # alpha, s11, s12, s21, s22, s23, q1
                    [2., 31., 32., 41., 42., 43., 4.],
                ],
                [],  # no seasonal
                None,
                [
                    [1., 3.],  # alpha, q1
                    [2., 4.],
                ],
            ],
            [  # replace seasonal
                dict(use_arma_errors=True, seasonal_periods=(2), p=1, q=1),
                [
                    [1., 11., 12., 3., 5.],  # alpha, s11, s2, p1, q1
                    [2., 31., 32., 4., 6.],
                ],
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                [3],
                [
                    [1., 0.1, 0.2, 0.3, 3., 5.],  # alpha, s1, s2, s3, p1, q1
                    [2., 0.4, 0.5, 0.6, 4., 6.],
                ],
            ],
            [  # with vector
                dict(use_arma_errors=True, seasonal_periods=(2), p=1, q=1),
                [
                    [1., 11., 12., 3., 5.],  # alpha, s11, s2, p1, q1
                ],
                [
                    [0.1, 0.2, 0.3],
                ],
                [3],
                [
                    [1., 0.1, 0.2, 0.3, 3., 5.],  # alpha, s1, s2, s3, p1, q1
                ],
            ],
            [  # should work with 1D vectors
                dict(use_arma_errors=True, seasonal_periods=(2), p=1, q=1),
                [1., 11., 12., 3., 5.],  # alpha, s11, s2, p1, q1
                [0.1, 0.2, 0.3],  # s1, s2, s3
                [3],
                [
                    [1., 0.1, 0.2, 0.3, 3., 5.],  # alpha, s1, s2, s3, p1, q1
                ],
            ],
            [  # remove seasonal, vector
                dict(use_arma_errors=True, seasonal_periods=(2, 3), q=1),
                [
                    [1., 11., 12., 21., 22., 23., 3.],  # alpha, s11, s12, s21, s22, s23, q1
                ],
                [],  # no seasonal
                None,
                [
                    [1., 3.],  # alpha, q1
                ],
            ],
        ]
    )
    def ktest_with_replaced_seasonal(self, components, matrix, new_seasonal_part, new_seasonal_periods,
                                     expected_matrix):
        components = Components(**components)
        matrix_obj = ComponentMatrix(matrix, components)
        replaced_seasonal = matrix_obj.with_replaced_seasonal(
            new_seasonal_part, new_seasonal_periods
        )
        assert np.array_equal(expected_matrix, replaced_seasonal.as_matrix())

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
                dict(seasonal_periods=(2, 3), use_arma_errors=True, q=1),
                [
                    [1., 11., 12., 21., 22., 23., 3.],  # alpha, s11, s12, s21, s22, s23, q1
                ],
                [
                    [1.],  # alpha
                ],
                [
                    [11., 12., 21., 22., 23.],  # seasonal
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
