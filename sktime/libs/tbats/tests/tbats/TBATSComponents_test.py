import pytest
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import Components

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSComponents(object):

    @pytest.mark.parametrize(
        "seasonal_periods, seasonal_harmonics, expected_harmonics",
        [
            [  # no periods means no harmonics
                None, None, [],
            ],
            [  # one period, one harmonic
                [5], [2], [2],
            ],
            [  # should initialize to harmonics with values of one
                [3, 4], None, [1, 1],
            ],
            [  # should work with float period lengths
                [3.2, 4.5], None, [1, 1],
            ],
            [  # should convert harmonics to int
                [3.2, 4.5], [1.2, 3.3], [1, 3],
            ],
        ]
    )
    def test_seasonal_harmonics(self, seasonal_periods, seasonal_harmonics, expected_harmonics):
        c = Components(seasonal_periods=seasonal_periods, seasonal_harmonics=seasonal_harmonics)
        assert np.array_equal(expected_harmonics, c.seasonal_harmonics)

        copied = c.with_arma(p=1, q=2)
        assert np.array_equal(expected_harmonics, copied.seasonal_harmonics)

    def test_with_seasonal_periods(self):
        old_periods = [3.3, 4.4]
        old_harmonics = [2, 1]
        c = Components(seasonal_periods=old_periods, seasonal_harmonics=old_harmonics)
        new_periods = [2, 3, 4.4]
        copied = c.with_seasonal_periods(seasonal_periods=new_periods)

        # old components should not change
        assert np.array_equal(old_periods, c.seasonal_periods)
        assert np.array_equal(old_harmonics, c.seasonal_harmonics)

        # new components should have new periods and ones for harmonics
        assert np.array_equal(new_periods, copied.seasonal_periods)
        assert np.array_equal([1, 1, 1], copied.seasonal_harmonics)

    def test_without_seasonal_periods(self):
        old_periods = [3.3, 4.4]
        old_harmonics = [2, 1]
        c = Components(seasonal_periods=old_periods, seasonal_harmonics=old_harmonics)
        copied = c.without_seasonal_periods()

        # old components should not change
        assert np.array_equal(old_periods, c.seasonal_periods)
        assert np.array_equal(old_harmonics, c.seasonal_harmonics)

        # new components should have no periods and no harmonics
        assert np.array_equal([], copied.seasonal_periods)
        assert np.array_equal([], copied.seasonal_harmonics)
