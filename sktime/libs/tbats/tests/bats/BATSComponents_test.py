import pytest
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats import Components

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestComponents(object):

    def test_period_normalization(self):
        periods = [2.2, 3.3, 1.0]
        c = Components(seasonal_periods=periods)
        assert np.array_equal([2, 3, 1], c.seasonal_periods)

    @pytest.mark.parametrize(
        "old_periods, new_periods",
        [
            [  # no periods before and after
                [], [],
            ],
            [  # replaced period
                [5], [2],
            ],
            [  # 1 period changed to 3
                [7], [3, 4, 6],
            ],
        ]
    )
    def test_with_seasonal_periods(self, old_periods, new_periods):
        c = Components(seasonal_periods=old_periods)
        copied = c.with_seasonal_periods(seasonal_periods=new_periods)

        # old components should not change
        assert np.array_equal(old_periods, c.seasonal_periods)

        # new components should have new periods and zeros for harmonics
        assert np.array_equal(new_periods, copied.seasonal_periods)

    def test_without_seasonal_periods(self):
        old_periods = [3, 4]
        c = Components(seasonal_periods=old_periods)
        copied = c.without_seasonal_periods()

        # old components should not change
        assert np.array_equal(old_periods, c.seasonal_periods)

        # new components should have no periods and no harmonics
        assert np.array_equal([], copied.seasonal_periods)
