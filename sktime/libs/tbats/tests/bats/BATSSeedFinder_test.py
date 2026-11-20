import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats.Components import Components
from sktime.libs.tbats.abstract.ComponentMatrix import ComponentMatrix
from sktime.libs.tbats.bats.SeedFinder import SeedFinder

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSSeedFinder(object):

    @pytest.mark.parametrize(
        "seasonal_periods, expected_mask",
        [
            [  # no periods means no mask
                [], [],
            ],
            [  # one period always produces a mask with one zero
                [6], [0],
            ],
            [  # two periods with no common divisor produce a mask of zeroes
                [3, 7], [0, 0],
            ],
            [  # If one period is a subperiod of the other, the mask contains 1 for the smaller period
                [3, 5, 6, 24], [1, 0, 1, 0],
            ],
            [  # If one period is a subperiod of the other, the mask contains 1 for the smaller period
                [2, 5, 15, 16], [1, 1, 0, 0],
            ],
            [  # If two periods have a common divisor then mask for the larger one contains this divisor
                [4, 6], [0, -2],
            ],
            [
                # If more than two periods have a common divisor then mask for the largest one contains divisor from smallest period
                [12, 42, 44], [0, -6, -4],  # -4 instead of -2
            ],
            [
                # If more than two periods have a common divisor then mask for the larger one contains divisor from smaller period
                [9, 16, 24], [0, 0, -3],  # -3 instead of -4
            ],
            [  # being a subperiod is more important than having a divisor
                [4, 6, 12], [1, 1, 0],
            ],
            [  # divisors and periods together
                [4, 5, 10, 14, 15], [0, 1, -2, -2, -5],
            ],
            [  # divisors and periods together
                [7, 9, 11, 12, 22, 30, 33], [0, 0, 1, -3, -2, -3, -3],
            ],
            [  # divisors and periods together
                [7, 9, 11, 12, 22, 30, 44], [0, 0, 1, -3, 1, -3, -4],
            ],
        ]
    )
    def test_prepare_mask(self, seasonal_periods, expected_mask):
        mask = SeedFinder.prepare_mask(seasonal_periods)
        assert np.array_equal(expected_mask, mask)

    @pytest.mark.parametrize(
        "seasonal_periods, w_tilda, expected",
        [
            [  # no periods means nothing to cut
                [], [1], np.zeros((0, 1)),
            ],
            [  # cut last param for each season
                [2],
                [
                    [1, 1, 2],  # alpha, 2 seasonal params
                    [1, 3, 4],
                ],
                [
                    [1],
                    [3],
                ],
            ],
            [  # two periods, one is a sub-period of the other, cut the sub-period out entirely
                [2, 4],
                [
                    [1, 1, 2, 0.1, 0.2, 0.3, 0.4],
                    [1, 3, 4, 0.5, 0.6, 0.7, 0.8],
                ],
                [
                    [0.1, 0.2, 0.3],
                    [0.5, 0.6, 0.7],
                ],
            ],
            [  # two periods with common divisor, cut out whole divisor from the larger one
                [4, 6],  # divisor = 2
                [
                    [1, 1.1, 2, 3.1, 4, 0.1, 0.2, 0.3, 0.41, 0.5, 0.6],
                    [1, 1.2, 2, 3.2, 4, 0.1, 0.2, 0.3, 0.42, 0.5, 0.6],
                    [1, 1.3, 2, 3.3, 4, 0.1, 0.2, 0.3, 0.43, 0.5, 0.6],
                ],
                [
                    [1.1, 2, 3.1, 0.1, 0.2, 0.3, 0.41],
                    [1.2, 2, 3.2, 0.1, 0.2, 0.3, 0.42],
                    [1.3, 2, 3.3, 0.1, 0.2, 0.3, 0.43],
                ],
            ],
            [  # period 2 has a common divisor with period 6, period 2 is a sub-period of 4 and 6
                [2, 4, 6],
                [
                    [1, 0.01, 0.02, 0.11, 0.2, 0.31, 0.4, 11, 2, 3, 41, 5, 6],
                    [1, 0.01, 0.02, 0.12, 0.2, 0.32, 0.4, 12, 2, 3, 42, 5, 6],
                ],
                [
                    [0.11, 0.2, 0.31, 11, 2, 3, 41],
                    [0.12, 0.2, 0.32, 12, 2, 3, 42],
                ],
            ],
        ]
    )
    def test_prepare_seasonal_params(self, seasonal_periods, w_tilda, expected):
        components = Components(seasonal_periods=seasonal_periods)
        w_tilda = ComponentMatrix(w_tilda, components)
        adj = SeedFinder(components)
        new_seasonal_params, _ = adj.prepare_seasonal_params(w_tilda)
        assert np.array_equal(expected, new_seasonal_params)

    @pytest.mark.parametrize(
        "component_params, w_tilda, expected",
        [
            [
                dict(),  # default params
                [
                    [1],  # 1 only
                    [1],
                ],
                [
                    [1],  # 1 only
                    [1],
                ],
            ],
            [
                dict(use_trend=True),
                [
                    [1, 1.1],  # 1, beta
                    [1, 2.1],
                    [1, 3.1],
                ],
                [
                    [1, 1.1],  # 1, beta
                    [1, 2.1],
                    [1, 3.1],
                ],
            ],
            [  # ARMA should not be included in linear regression
                dict(use_arma_errors=True, p=2, q=1),
                [
                    [1, 1.1, 1.2, 1.3],  # 1, p1, p2, q1
                    [2, 2.1, 2.2, 2.3],
                    [3, 3.1, 3.2, 3.3],
                ],
                [
                    [1],  # 1,
                    [2],
                    [3],
                ],
            ],
            [  # one season, should simply remove last seasonal parameter
                dict(seasonal_periods=(3)),
                [
                    [1, 1.1, 1.2, 1.3],  # 1, s1, s2, s3
                ],
                [
                    [1, 1.1, 1.2],  # 1, s1, s2
                ],
            ],
            [  # two seasons, where one is a sub-season of the other, sub-season should be removed from params
                dict(seasonal_periods=(2, 4)),
                [
                    [1, 1.1, 1.2, 2.1, 2.2, 2.3, 2.4],  # 1, s11, s12, s21, s22, s23, s24
                ],
                [
                    [1, 2.1, 2.2, 2.3],  # 1, s21, s22, s23
                ],
            ],
            [  # two seasons, where one has a common divisor of 2 with the other,
                # params for longer period should be shrunk by the divisor=2
                dict(seasonal_periods=(4, 6)),
                [
                    [1, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
                ],
                [
                    [1, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4],
                ],
            ],
        ]
    )
    def test_conversions(self, component_params, w_tilda, expected):
        components = Components(**component_params)

        converter = SeedFinder(components)
        matrix = converter.to_matrix_for_linear_regression(w_tilda)
        assert np.array_equal(expected, matrix)

    @pytest.mark.parametrize(
        "component_params, lr_coefs, expected_x0",
        [
            [
                dict(),  # default params
                [1.],
                [1.],
            ],
            [  # ARMA coefs should be added back as zeros
                dict(use_trend=True, use_damped_trend=True, use_arma_errors=True, p=1, q=2),
                [0.6, 0.5],  # alpha, beta
                [0.6, 0.5, 0., 0., 0.],
            ],
            [  # Parameters for single season should sum to 0, last param should be added back
                dict(seasonal_periods=(3)),
                [0.6, 2, 4],  # alpha, s1, s2
                [0.6, 0, 2, -2],  # alpha, s1, s2, s3
            ],
            [  # When one season is a sub-season of other it should be added back as zeros
                dict(seasonal_periods=(3, 6)),
                [0.6, 2, 4, 6, 8, 10],  # alpha, s21, s22, ..., s25
                [0.6, 0, 0, 0, -3, -1, 1, 3, 5, -5],  # alpha, s11, s12, s13, s21, ..., s25, s26
            ],
            [  # When there is a divisor of 2 between two seasons, the longer season should receive 2 params back
                dict(seasonal_periods=(4, 6)),
                [0.6, 2, 4, 6, 1, 2, 3, 6],  # alpha, s11, s12, s13, s21, s22, s23, ..., s24
                [0.6, -1, 1, 3, -3, -1, 0, 1, 4, -2, -2],  # alpha, s11, s12, s13,, s14, s21, ..., s25, s26
            ],
        ]
    )
    def test_back_conversions(self, component_params, lr_coefs, expected_x0):
        components = Components(**component_params)

        converter = SeedFinder(components)
        x0 = converter.from_linear_regression_coefs_to_x0(lr_coefs)
        assert np.array_equal(expected_x0, x0)
