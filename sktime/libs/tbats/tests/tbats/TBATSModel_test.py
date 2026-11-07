import numpy as np
import pytest
import scipy.stats as stats
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import Components, ModelParams, Model, Context

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSModel(object):

    def create_model(self, params):
        return Model(params, Context())

    def test_fit_alpha_only(self):
        alpha = 0.7
        np.random.seed(345)
        T = 200

        l = l0 = 0.2
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        c = Components(use_arma_errors=False)
        p = ModelParams(c, alpha=alpha, x0=np.array([l0]))
        model = self.create_model(p)
        fitted_model = model.fit(y)
        resid = fitted_model.resid

        # Residuals should form a normal distribution
        _, pvalue = stats.normaltest(resid)
        assert 0.05 < pvalue  # large p-value, we can not reject null hypothesis of normal distribution

        # Mean of residuals should be close to 0
        _, pvalue = stats.ttest_1samp(resid, popmean=0.0)
        assert 0.05 < pvalue  # large p-value we can not reject null hypothesis that mean is 0

        # We expect 95% of residuals to lie within [-2,2] interval
        assert len(resid[np.where(np.abs(resid) < 2)]) / len(resid) > 0.90

    @pytest.mark.parametrize(
        "seasonal_periods, seasonal_harmonics, starting_values",
        [
            [
                [4], [1], [[2, 0]],  # s1, s1*
            ],
            [
                [365], [2], [[1, 2, 0.5, 0.6]]  # s1, s2, s1*, s2*
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
        The aim of the test is to check if model is correctly calculating trigonometric series with no noise.
        """
        T = 60
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

        c = Components(use_arma_errors=False, seasonal_periods=seasonal_periods, seasonal_harmonics=seasonal_harmonics)
        p = ModelParams(c, alpha=0.0, x0=np.array(x0))  # gamma_params initialize to zeros here
        model = self.create_model(p)
        fitted_model = model.fit(y_to_fit)
        resid = fitted_model.resid

        # sequence should be modelled perfectly
        assert np.allclose([0] * (T - steps), resid)
        assert np.allclose(y_to_fit, fitted_model.y_hat)

        # forecast should be perfect
        y_predicted = fitted_model.forecast(steps=steps)
        assert np.allclose(y_to_predict, y_predicted)

    def test_fit_alpha_and_trigonometric_series(self):
        alpha = 0.7
        gamma1 = 0.1
        gamma2 = 0.05
        period_length = 4
        np.random.seed(2342)
        T = 300

        l = l0 = 0.2
        s = s0 = 1
        s_star = s0_star = 0
        y = [0] * T
        lam = 2 * np.pi / period_length
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + s + d
            l = l + alpha * d
            s_prev = s
            s = s_prev * np.cos(lam) + s_star * np.sin(lam) + gamma1 * d
            s_star = - s_prev * np.sin(lam) + s_star * np.cos(lam) + gamma2 * d

        c = Components(use_arma_errors=False, seasonal_periods=[period_length], seasonal_harmonics=[1])
        p = ModelParams(c, alpha=alpha,  # gamma_params=[gamma1, gamma2],
                        x0=np.array([l0, s0, s0_star]))
        model = self.create_model(p)
        fitted_model = model.fit(y)
        resid = fitted_model.resid

        # Residuals should form a normal distribution
        _, pvalue = stats.normaltest(resid)
        assert 0.05 < pvalue  # large p-value, we can not reject null hypothesis of normal distribution

        # Mean of residuals should be close to 0
        _, pvalue = stats.ttest_1samp(resid, popmean=0.0)
        assert 0.05 < pvalue  # large p-value we can not reject null hypothesis that mean is 0

        # We expect 95% of residuals to lie within [-2,2] interval
        # 0.8 - lets choose something that expected 0.95 will meet
        assert len(resid[np.where(np.abs(resid) < 2)]) / len(resid) > 0.80

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

        c = Components(use_arma_errors=False, use_trend=True, use_damped_trend=True,
                       seasonal_periods=[period_length], seasonal_harmonics=[1])
        p = ModelParams(c, phi=phi,
                        # alpha, beta, gamma_params do not matter in this case as there is no noise
                        alpha=0, beta=0, gamma_params=[0, 0],
                        x0=np.array([l0, b0, s0, s0_star]))
        model = self.create_model(p)
        fitted_model = model.fit(y_to_fit)
        resid = fitted_model.resid

        # sequence should be modelled perfectly
        assert np.allclose([0] * (T - steps), resid)
        assert np.allclose(y_to_fit, fitted_model.y_hat)

        # forecast should be perfect
        y_predicted = fitted_model.forecast(steps=steps)
        assert np.allclose(y_to_predict, y_predicted)

    def test_forecast_confidence_intervals(self):

        T = 30
        steps = 5

        period_length = 6
        y = [0] * T
        b = b0 = 2.1
        l = l0 = 1.2
        alpha = 0.5
        beta = 0.2
        np.random.seed(3433)
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + b + d
            l = l + b + alpha * d
            b = b + beta * d

        c = Components(use_arma_errors=False, use_trend=True, use_damped_trend=False, use_box_cox=False)
        p = ModelParams(c, alpha=0.5, beta=0.2, x0=[1.2, 2.1])
        model = self.create_model(p)
        model = model.fit(y)
        forecasts, confidence_info = model.forecast(steps=4, confidence_level=0.95)
        assert 0.95 == confidence_info['calculated_for_level']
        assert np.array_equal(forecasts, confidence_info['mean'])
        assert np.allclose([59.46379894, 60.44336595, 61.290095, 62.02915299], confidence_info['lower_bound'])
        assert np.allclose([62.99372071, 64.75218458, 66.64348641, 68.6424593], confidence_info['upper_bound'])
