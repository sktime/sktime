import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import Components, ModelParams

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSModelParams(object):

    @pytest.mark.parametrize(
        "components, params, expected_gamma, expected_gamma_1, expected_gamma_2",
        [
            [  # no seasonal periods, empty gammas
                dict(),
                dict(alpha=1),
                [], [], [],
            ],
            [  # one seasonal period
                dict(seasonal_periods=[4.2]),
                dict(alpha=1, gamma_params=[0.5, 1.1]),
                [0.5, 1.1], [0.5], [1.1],
            ],
            [  # 3 periods, 3 coefs in each gamma vector
                dict(seasonal_periods=[3, 4, 5]),
                dict(alpha=1, gamma_params=[1.1, 1.2, 2.1, 2.2, 3.1, 3.2]),
                [1.1, 1.2, 2.1, 2.2, 3.1, 3.2], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2],
            ],
            [  # should initialize gamma to zeros
                dict(seasonal_periods=[3, 4, 5]),
                dict(alpha=1, beta=0.2),
                [0.] * 6, [0., 0., 0.], [0., 0., 0.],
            ],
        ]
    )
    def test_gamma(self, components, params, expected_gamma, expected_gamma_1, expected_gamma_2):
        c = Components(**components)
        p = ModelParams(c, **params)

        assert np.array_equal(expected_gamma, p.gamma_params)
        assert np.array_equal(expected_gamma_1, p.gamma_1())
        assert np.array_equal(expected_gamma_2, p.gamma_2())
