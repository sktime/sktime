import pytest
import numpy as np
import multiprocessing
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
import sktime.libs.tbats.error as error
from sktime.libs.tbats import TBATS

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATS(object):

    def test_constant_model(self):
        y = [3.2] * 20
        estimator = TBATS()
        model = estimator.fit(y)
        assert np.allclose([0.0] * len(y), model.resid)
        assert np.allclose(y, model.y_hat)
        assert np.allclose([3.2] * 5, model.forecast(steps=5))

    def test_normalize_seasonal_periods(self):
        seasonal_periods = [7, 0, 1, 9, 9, 8.8, 10.11, 3, -1, 2, 1.01]
        with pytest.warns(error.InputArgsWarning):
            estimator = TBATS(seasonal_periods=seasonal_periods)
        # seasonal periods should be normalized in constructor
        # seasonal periods should be greater than 1, unique and sorted
        assert np.array_equal([1.01, 2, 3, 7, 8.8, 9, 10.11], estimator.seasonal_periods)

    @pytest.mark.parametrize(
        "definition, expected_components",
        [
            [  # default settings allow for all components
                dict(),
                dict(use_box_cox=True, use_trend=True, use_damped_trend=True,
                     seasonal_periods=[], seasonal_harmonics=[]),
            ],
            [  # default settings allow for all components
                dict(use_box_cox=False, use_trend=False),
                dict(use_box_cox=False, use_trend=False, use_damped_trend=False,
                     seasonal_periods=[], seasonal_harmonics=[]),
            ],
            [  # default settings allow for all components
                dict(use_box_cox=True, use_damped_trend=False, seasonal_periods=[7, 31]),
                dict(use_box_cox=True, use_trend=True, use_damped_trend=False,
                     seasonal_periods=[7, 31], seasonal_harmonics=[1, 1]),
            ],
        ]
    )
    def test_create_most_complex_components(self, definition, expected_components):
        estimator = TBATS(**definition)
        components = estimator.create_most_complex_components()

        # ARMA is false as it will be used in the end, once harmonics were chosen
        assert False == components.use_arma_errors

        assert expected_components['use_box_cox'] == components.use_trend
        assert expected_components['use_trend'] == components.use_trend
        assert expected_components['use_damped_trend'] == components.use_damped_trend

        assert np.array_equal(expected_components['seasonal_periods'], components.seasonal_periods)
        assert np.array_equal(expected_components['seasonal_harmonics'], components.seasonal_harmonics)

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(),
        reason="multiprocessing start method 'fork' is unavailable",
    )
    def test_trend_and_seasonal(self):
        T = 30
        steps = 5

        phi = 0.99
        period_length = 6
        y = [0] * T
        b = b0 = 2.1
        l = l0 = 1.2
        s = s0 = 0
        s_star = s0_star = 0.2
        for t in range(0, T):
            y[t] = l + phi * b + s
            l = l + phi * b
            b = phi * b
            lam = 2 * np.pi / period_length
            s_prev = s
            s = s_prev * np.cos(lam) + s_star * np.sin(lam)
            s_star = - s_prev * np.sin(lam) + s_star * np.cos(lam)

        y_to_fit = y[:(T - steps)]
        y_to_predict = y[(T - steps):]

        # pytest does not work well with spawn multiprocessing method
        # https://github.com/pytest-dev/pytest/issues/958
        estimator = TBATS(use_arma_errors=False, use_trend=True, use_damped_trend=True, use_box_cox=False,
                          seasonal_periods=[period_length], multiprocessing_start_method='fork')

        fitted_model = estimator.fit(y_to_fit)
        resid = fitted_model.resid

        # seasonal model with 1 harmonic should be chosen
        assert np.array_equal([1], fitted_model.params.components.seasonal_harmonics)
        assert np.array_equal([period_length], fitted_model.params.components.seasonal_periods)

        assert np.isclose(phi, fitted_model.params.phi, atol=0.01)

        # from some point residuals should be close to 0
        assert np.allclose([0] * (T - steps - 10), resid[10:], atol=0.06)
        assert np.allclose(y_to_fit[10:], fitted_model.y_hat[10:], atol=0.06)

        # forecast should be close to actual sequence
        y_predicted = fitted_model.forecast(steps=steps)
        assert np.allclose(y_to_predict, y_predicted, atol=0.5)

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(),
        reason="multiprocessing start method 'fork' is unavailable",
    )
    @pytest.mark.parametrize(
        "seasonal_periods, seasonal_harmonics, starting_values",
        [
            [
                [12], [2], [[1, 2, 0.5, 0.6]]  # s1, s2, s1*, s2*
            ],
            [
                [7, 365], [2, 3], [[1, 2, 0.5, 0.6], [0.5, 0.2, 0.4, 0.1, 0.9, 0.3]]
            ],
            [  # non-integer period lengths should also work
                [7.2, 12.25], [2, 1], [[0.4, 0.7, 0.2, 0.1], [0.9, 0.8]]
            ],
            [  # 3 periods
                [7, 11, 13.2], [2, 4, 3],
                [[1, 2, 0.5, 0.6], [0.5, 0.2, 0.4, 0.1, 0.9, 0.3, 1.1, 1.2], [-0.1, 0.2, 0.7, 0.6, 0.3, -0.3]]
            ],
        ]
    )
    def test_fit_predict_trigonometric_seasonal(self, seasonal_periods, seasonal_harmonics, starting_values):
        """
        The aim of the test is to check if model is correctly discovering trigonometric series with no noise
        """
        T = 100
        steps = 10
        l = 3.1
        x0 = [[l]]

        # construct trigonometric series
        y = [l] * T
        for period in range(0, len(seasonal_periods)):
            period_length = seasonal_periods[period]
            period_harmonics = seasonal_harmonics[period]
            s_harmonic = np.array(starting_values[period])
            s = s_harmonic[:int(len(s_harmonic) / 2)]
            s_star = s_harmonic[int(len(s_harmonic) / 2):]
            x0.append(s_harmonic)
            lambdas = 2 * np.pi * (np.arange(1, period_harmonics + 1)) / period_length
            # add periodic impact to y
            for t in range(0, T):
                y[t] += np.sum(s)
                s_prev = s
                s = s_prev * np.cos(lambdas) + s_star * np.sin(lambdas)
                s_star = - s_prev * np.sin(lambdas) + s_star * np.cos(lambdas)

        x0 = np.concatenate(x0)

        y_to_fit = y[:(T - steps)]
        y_to_predict = y[(T - steps):]

        # pytest does not work well with spawn multiprocessing method
        # https://github.com/pytest-dev/pytest/issues/958
        estimator = TBATS(use_box_cox=False, use_arma_errors=False, use_trend=False,
                          seasonal_periods=seasonal_periods,
                          multiprocessing_start_method='fork')
        fitted_model = estimator.fit(y_to_fit)
        resid = fitted_model.resid

        # seasonal model should be discovered
        assert np.array_equal(seasonal_periods, fitted_model.params.components.seasonal_periods)
        # at least as many harmonics as in original series
        assert np.all(np.asarray(seasonal_harmonics) <= fitted_model.params.components.seasonal_harmonics)

        # sequence should be modelled properly
        assert np.allclose([0] * (T - steps), resid, atol=0.2)
        assert np.allclose(y_to_fit, fitted_model.y_hat, atol=0.2)

        # forecast should be close to actual
        y_predicted = fitted_model.forecast(steps=steps)
        assert np.allclose(y_to_predict, y_predicted, 0.2)

    def test_no_numeric_regression_issues(self):
        """
        Test https://github.com/intive-DataScience/tbats/issues/40.

        The issue was caused by numeric precision issue for harmonics that start as 0 but are regressed
        to absurd high values.
        """
        y = [
            4140, 4510, 4378, 5010, 5222, 6260, 5094, 5854, 6010, 6428, 5890, 6414,
            6346, 6034, 6770, 7450, 7266, 6788, 6162, 8476, 6480, 5872, 5810, 6522,
            6444, 6110, 5778, 6068, 6018, 6174, 5202, 4820, 5114, 5686, 4946,
        ]
        estimator = TBATS(seasonal_periods=[2], n_jobs=1, use_box_cox=True)
        _ = estimator.fit(y) # should not fail
