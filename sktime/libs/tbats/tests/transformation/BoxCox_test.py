import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
import sktime.libs.tbats.transformation as transformation

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBoxCox(object):

    @pytest.mark.parametrize(
        "y, lam, expected_y",
        [
            [  # lambda is zero, expect logarithm
                [np.e, np.e ** 2, np.e ** 3],
                0,
                [1., 2., 3.],
            ],
            [  # lambda is 1, expect simple addition
                [-2, -1, 0, 1, 2, 3],
                1,
                [-3, -2, -1, 0, 1, 2],
            ],
            [  # lambda is 0.5, expect sqrt
                [4, 9, 16, -4, -9],
                0.5,
                [(2 - 1) / 0.5, (3 - 1) / 0.5, (4 - 1) / 0.5, -(2 + 1) / 0.5, -(3 + 1) / 0.5],
            ],
            [  # lambda is -2, expect inversion and power of 2
                [0.1, 1, 3],
                -2,
                [(100 - 1) / (-2), 0, (1 / 9 - 1) / (-2)],
            ],
        ]
    )
    def test_transformations(self, y, lam, expected_y):
        y_transformed = transformation.boxcox(y, lam)
        assert np.allclose(expected_y, y_transformed)
        y_back_transformed = transformation.inv_boxcox(y_transformed, lam)
        assert np.allclose(y, y_back_transformed)

    def test_lambda_finding(self):
        np.random.seed(49385)
        y = np.exp(
            np.random.uniform(size=100) + np.array(range(40, 140)) / 20
        )
        lam = transformation.find_box_cox_lambda(y, bounds=(0, 1))
        assert np.isclose(0.031759, lam, atol=1e-4)

    def test_lambda_finding_2(self):
        np.random.seed(49385)
        y = (np.random.uniform(size=105) + np.array(range(40, 145)) / 20) ** 2
        lam = transformation.find_box_cox_lambda(y, bounds=(0, 1))
        assert np.isclose(0.4139901, lam, atol=1e-4)

    def test_lambda_finding_with_seasonality(self):
        np.random.seed(3423)
        t = np.array(range(0, 160))
        y = 5 * np.sin(t * 2 * np.pi / 14) + ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10

        lam = transformation.find_box_cox_lambda(y, seasonal_periods=[9, 14], bounds=(0, 1))
        assert np.isclose(0.690025, lam, atol=1e-4)
