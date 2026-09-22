import pytest
import numpy as np
import multiprocessing
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
import sktime.libs.tbats.error as error
from sktime.libs.tbats import BATS

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATS(object):

    def test_input_validation(self):
        estimator = BATS()
        with pytest.raises(error.InputArgsException):
            estimator.fit([])
        with pytest.raises(error.InputArgsException):
            estimator.fit('string')

    def test_seasonal_periods_input_validation(self):
        with pytest.warns(error.InputArgsWarning):
            BATS(seasonal_periods=[1, 3])
        with pytest.warns(error.InputArgsWarning):
            BATS(seasonal_periods=[0, 3])

    def test_constant_model(self):
        y = [2.9] * 20
        estimator = BATS()
        model = estimator.fit(y)
        assert np.allclose([0.0] * len(y), model.resid)
        assert np.allclose(y, model.y_hat)
        assert np.allclose([2.9] * 5, model.forecast(steps=5))

    @pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="multiprocessing start method 'fork' is unavailable",
    )
    def test_fit_only_alpha(self):
        alpha = 0.8
        np.random.seed(333)
        T = 200

        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        # pytest does not work well with spawn multiprocessing method
        # https://github.com/pytest-dev/pytest/issues/958
        estimator = BATS(multiprocessing_start_method='fork')
        model = estimator.fit(y)
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
