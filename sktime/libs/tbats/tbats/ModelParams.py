import numpy as np

from ..abstract import ArrayHelper, ModelParams as AbstractModelParams


class ModelParams(AbstractModelParams):
    """Holds all parameters needed to calculate TBATS model and forecasts

    For other parameters documentation see parent class documentation.

    Parameters
    ----------
    gamma_params: array-like of floats
        Gamma smoothing parameters for seasonal effects.
        In TBATS there are two smoothing parameters for each season.
        Vector holds 2 gamma parameters for season 1, then 2 gamma parameters for season 2, ...
    """

    def __init__(self, components, alpha, beta=None, phi=None, box_cox_lambda=None,
                 gamma_params=None,
                 ar_coefs=None, ma_coefs=None, x0=None):
        """See parent class documentation for details"""
        super().__init__(components=components, alpha=alpha, beta=beta, phi=phi, box_cox_lambda=box_cox_lambda,
                         gamma_params=gamma_params, ar_coefs=ar_coefs, ma_coefs=ma_coefs, x0=x0)

    @classmethod
    def with_default_starting_params(cls, y, components):
        """Factory method for starting model parameters.

        Parameters
        ----------
        y: array-like of floats
            Time series
        components: Components
            Components used in the model

        Returns
        -------
        ModelParams
            Default starting params.
        """
        alpha = 0.09

        beta = None
        phi = None
        if components.use_trend:
            beta = 0.05
            phi = 1
            if components.use_damped_trend:
                phi = 0.999

        # Here TBATS differs from BATS as staring gamma parameters are zeroes for TBATS
        gamma_params = None
        if len(components.seasonal_periods) > 0:
            gamma_params = [0.0] * components.gamma_params_amount()

        box_cox_lambda = cls.find_initial_box_cox_lambda(y, components)

        # note that ARMA and x0 will be initialized to 0's in constructor
        return cls(components=components,
                   alpha=alpha, beta=beta, phi=phi,
                   gamma_params=gamma_params,
                   box_cox_lambda=box_cox_lambda)

    def seasonal_components_amount(self):
        """TBATS requires this many seed state values for seasonalities"""
        return int(2 * self.components.seasonal_harmonics.sum())

    def gamma_1(self):
        """Gamma 1 smoothing parameters are kept in even position of gamma vector"""
        return self.gamma_params[range(0, len(self.gamma_params), 2)]

    def gamma_2(self):
        """Gamma 2 smoothing parameters are kept in odd position of gamma vector"""
        return self.gamma_params[range(1, len(self.gamma_params), 2)]

    def _normalize_gamma_params(self, gamma_params=None):
        """Ensures gamma_params is an array. If necessary it is initialized to zeroes."""
        gamma_params = ArrayHelper.to_array(gamma_params, float)
        expected_params_amount = 2 * len(self.components.seasonal_periods)
        if len(gamma_params) != expected_params_amount:
            gamma_params = np.asarray([0.0] * expected_params_amount, float)
        return gamma_params
