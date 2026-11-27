import numpy as np
import pytest
import scipy.stats as stats
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.bats.Components import Components
from sktime.libs.tbats.bats.Context import Context
from sktime.libs.tbats.bats.Model import Model
from sktime.libs.tbats.bats.ModelParams import ModelParams

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestBATSModel(object):

    def create_model(self, params):
        return Model(params, Context())

    @pytest.mark.parametrize(
        "components, params, expected_can_be_admissible, expected_is_admissible",
        [
            [
                dict(),
                dict(alpha=0.5),
                True,
                True,
            ],
            [  # D matrix eigen values check should fail
                dict(),
                dict(alpha=2.1),
                False,
                False,
            ],
            [
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=0.4, phi=0.8),
                True,
                True,
            ],
            [  # phi check should fail
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=0.4, phi=1.01),
                False,
                False,
            ],
            [  # phi check should fail
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.2, beta=0.4, phi=0.79),
                False,
                False,
            ],
            [  # box cox lambda check should fail
                dict(use_box_cox=True),
                dict(alpha=0.3, box_cox_lambda=2.0),
                False,
                False,
            ],
            [  # box cox lambda check should fail
                dict(use_box_cox=True, box_cox_bounds=(-1, 0)),
                dict(alpha=0.1, box_cox_lambda=0.5),
                False,
                False,
            ],
            [  # D matrix eigen values check should fail
                dict(use_trend=True, use_damped_trend=True),
                dict(alpha=0.5, beta=3.4, phi=0.9),
                False,
                False,
            ],
            [
                dict(use_arma_errors=True, p=2, q=0),
                dict(alpha=0.5, ar_coefs=[0.5, 0.2]),
                True,
                True,
            ],
            [  # AR is not stationary
                dict(use_arma_errors=True, p=2, q=0),
                dict(alpha=0.5, ar_coefs=[2, 3]),
                False,
                False,
            ],
            [
                dict(use_arma_errors=True, p=0, q=2),
                dict(alpha=0.5, ma_coefs=[0.4, 0.2]),
                True,
                True,
            ],
            [  # MA is not invertible
                dict(use_arma_errors=True, p=0, q=2),
                dict(alpha=0.5, ma_coefs=[2, 3]),
                False,
                False,
            ],
        ]
    )
    def test_admissibility(self, components, params, expected_can_be_admissible, expected_is_admissible):
        model_params = ModelParams(Components(**components), **params)
        np.random.seed(123)
        y = np.random.normal(size=(30))
        model = self.create_model(model_params)
        assert not model.is_fitted
        assert not model.is_admissible()
        assert expected_can_be_admissible == model.can_be_admissible()
        if expected_can_be_admissible:
            model = model.fit(y)
            assert model.is_fitted
            assert expected_is_admissible == model.is_admissible()

    def test_constant_model(self):
        y = [5.3, 5.3, 5.3, 5.3, 5.3, ]
        c = Components(use_arma_errors=False)
        p = ModelParams(c, alpha=0, x0=[5.3])
        model = self.create_model(p).fit(y)
        assert np.allclose([0.0] * len(y), model.resid)
        assert np.allclose(y, model.y_hat)
        assert np.allclose([5.3] * 3, model.forecast(steps=3))

    @pytest.mark.parametrize(
        "components, params, x0, expected_y, expected_resid",
        [
            [
                dict(),
                dict(alpha=0.5),
                [0],
                [0, 1, 2.5],
                [2, 3, -0.5],
            ],
            [
                dict(use_trend=True),
                dict(alpha=0.5, beta=1),
                [0, 0],
                [0, 3, 6.5],
                [2, 1, -4.5],
            ],
            [  # the same result as for test before, phi=1 when damping is not used
                dict(use_trend=True),
                dict(alpha=0.5, beta=1, phi=1),
                [0, 0],
                [0, 3, 6.5],
                [2, 1, -4.5],
            ],

        ]
    )
    def test_fit(self, components, params, x0, expected_y, expected_resid):
        y = [2.0, 4.0, 2.0]
        c = Components(**components)
        p = ModelParams(c, x0=np.array(x0), **params)
        model = self.create_model(p)
        result = model.fit(y)
        assert np.allclose(expected_y, result.y_hat)
        assert np.allclose(expected_resid, result.resid)

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
        "components, params, expected_y, expected_resid, expected_aic",
        [
            [
                dict(),
                dict(alpha=0.5),
                [0., 1., 2.5],
                [2., 3., -0.5],
                11.751992657296693,
            ],
            [
                dict(use_trend=True),
                dict(alpha=0.5, beta=1),
                [0., 3., 6.5],
                [2., 1., -4.5],
                17.686478467164108,
            ],
        ]
    )
    def test_fit(self, components, params, expected_y, expected_resid, expected_aic):
        y = [2.0, 4.0, 2.0]
        c = Components(**components)
        p = ModelParams(c, **params)  # note x0 = 0 vector
        model = self.create_model(p)
        fitted_model = model.fit(y)
        assert fitted_model.is_fitted
        assert np.allclose(expected_y, fitted_model.y_hat)
        assert np.allclose(expected_resid, fitted_model.resid)
        assert np.isclose(expected_aic, fitted_model.aic)

    def test_fit_simple_seasonality_no_noise(self):
        season_length = 4
        seasonal_starting_params = [-0.5, 0.5, -0.3, 0.3]  # sum is 0
        np.random.seed(32234)
        T = 20

        l = 1
        y = [0] * T
        s = [0] * (T + season_length)
        s[0:season_length] = seasonal_starting_params
        for t in range(0, T):
            s[t + season_length] = s[t]
            y[t] = l + s[t + season_length]

        # seasonal starting params are in reverse order
        x0 = np.concatenate([[1], np.flip(seasonal_starting_params, axis=0)])

        params = ModelParams(
            Components(seasonal_periods=(4), use_arma_errors=False),
            alpha=0.5,  # the value of alpha does not matter in this sequence
            gamma_params=np.array([0.7]),  # the value of gamma does not matter in this sequence
            x0=x0
        )
        model = self.create_model(params)
        model.fit(y)

        # model should produce exactly the same result as input sequence
        assert np.array_equal(y, model.y_hat)

    def test_fit_with_arma(self):
        y = [
            2.00998628, 2.01834175, 1.73629466, 0.15158333,
            1.76079542, -1.51235184, 4.43603598, 2.39385310,
            -1.11931777, 2.47476369, 2.75227321, -1.23175380,
        ]
        expected_y_hat = [
            1.140684548, 1.022609210, 1.067214446, 1.325473037,
            1.484324145, 1.110343637, 1.417342489, 0.928206154,
            0.541250335, 2.391444376, 1.237958168, -0.031981245,
        ]
        params = ModelParams(
            Components(use_arma_errors=True, p=2, q=2),
            alpha=0.06897393,
            ar_coefs=[-0.9221481, -0.9506828],
            ma_coefs=[0.7173464, 0.7780436],
            x0=[1.140685, 0, 0, 0, 0],
        )
        model = self.create_model(params)
        model.fit(y)
        assert np.allclose(expected_y_hat, model.y_hat)

    def test_fit_with_boxcox(self):
        y = [
            11.79172712, 10.85108536, 15.3251649, 15.66841278, 14.43531692, 11.63829957, 26.40812776, 11.88435592,
            14.09793616, 26.06597912, 20.58004644, 24.34577183, 32.04387106, 17.47443736, 30.62936647, 30.72851384,
            28.94948062, 32.24077089, 22.43946496, 19.91517497, 30.64887888, 30.18141265, 25.68003731, 28.81030685,
            50.95861768, 56.93824159, 50.58329804, 33.02249249, 36.38440367, 55.87165424, 34.87575997,
        ]
        expected_y_hat = [
            18.38448392, 19.9224371, 21.32829308, 22.68290723, 23.94571775, 25.02424428, 25.75399778, 26.51833154,
            26.87053586, 26.87843899, 26.86984482, 26.71836618, 26.51837513, 26.4197001, 26.10467938, 25.8760056,
            25.73765624, 25.66032727, 25.70030341, 25.67058672, 25.51104836, 25.44570822, 25.46732205, 25.49319482,
            25.58160106, 26.02657242, 26.89713199, 28.15016821, 29.55579133, 31.16087487, 33.23902918
        ]
        params = ModelParams(
            Components(use_trend=True, use_damped_trend=False, use_arma_errors=False, use_box_cox=True),
            alpha=0.0,
            beta=0.02,
            box_cox_lambda=0,  # applies np.log
            x0=[2.8222853209364787, 0.08922172250433948],
        )
        model = self.create_model(params)
        model.fit(y)

        assert model.params.box_cox_lambda == 0
        assert np.allclose(expected_y_hat, model.y_hat)
        assert np.allclose(np.array(y) - np.array(expected_y_hat), model.resid)
        assert not np.allclose(model.resid, model.resid_boxcox)

    def test_forecast_with_trend(self):
        alpha = 0.8
        beta = 0.4
        np.random.seed(123)
        T = 20

        b = b0 = 2.2
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + b + d
            l = l + b + alpha * d
            b = b + beta * d

        horizon = 20
        y_to_forecast = [0] * T
        for t in range(0, horizon):
            y_to_forecast[t] = l + b
            l = l + b

        params = ModelParams(
            Components(use_trend=True, use_damped_trend=False, use_arma_errors=False, use_box_cox=False),
            alpha=alpha,
            beta=beta,
            x0=[l, b0],
        )
        model = self.create_model(params)
        model.fit(y)

        assert np.allclose(y_to_forecast, model.forecast(horizon))
