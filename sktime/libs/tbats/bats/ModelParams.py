import numpy as np

from ..abstract import ArrayHelper, ModelParams as AbstractModelParams


class ModelParams(AbstractModelParams):
    """Holds all parameters needed to calculate BATS model and forecasts

        For other parameters documentation see parent class documentation.

        Parameters
        ----------
        gamma_params: array-like of floats
            Gamma smoothing parameters for seasonal effects.
            In BATS there is one gamma parameter for each season.
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
        if components.seasonal_periods.sum() > 16:
            alpha = 1e-6

        beta = None
        phi = None
        if components.use_trend:
            beta = 0.05
            if components.seasonal_periods.sum() > 16:
                beta = 5e-7
            phi = 1
            if components.use_damped_trend:
                phi = 0.999

        gamma_params = None
        if len(components.seasonal_periods) > 0:
            gamma_params = [0.001] * components.gamma_params_amount()

        box_cox_lambda = cls.find_initial_box_cox_lambda(y, components)

        # note that ARMA and x0 will be initialized to 0's in constructor
        return cls(components=components,
                   alpha=alpha, beta=beta, phi=phi,
                   gamma_params=gamma_params,
                   box_cox_lambda=box_cox_lambda)

    def seasonal_components_amount(self):
        """BATS model requires this many seasonal seed state values"""
        return int(self.components.seasonal_periods.sum())

    def _normalize_gamma_params(self, gamma_params=None):
        """Ensures gamma params is an array. If necessary initializes Gamma params to zeroes"""
        gamma_params = ArrayHelper.to_array(gamma_params)
        if len(gamma_params) != len(self.components.seasonal_periods):
            gamma_params = np.asarray([0.0] * len(self.components.seasonal_periods), float)
        return gamma_params
