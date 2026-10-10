import copy
import numpy as np

from .. import transformation
from . import ArrayHelper


class ModelParams(object):
    """Holds all parameters needed to calculate model and forecasts

    Attributes
    ----------
    components: Components
        Defines which components are used in the model (trend, box-cox, seasonalities, ...)
    alpha: float
        Smoothing parameter value
    beta: float or None
        Trend smoothing parameter value.
        None when trend is not used.
        Also see components.use_trend.
    phi: float or None
        Trend damping parameter value.
        None when trend is not used. 1.0 when trend is used but damping is not.
        Also see components.use_damped_trend.
    box_cox_lambda: float or None
        Box-Cox transformation lambda parameter value
        None when series is not being transformed.
        Also see components.use_box_cox.
    gamma_params: array-like of floats
        Seasonal smoothing parameters.
        Empty array when there is no seasonality in the model.
        Also see components.seasonal_periods
    ar_coefs: array-like of floats
        AR(p) parameters used for residuals modeling.
        Empty array when there is no ARMA residuals modelling.
        Also see components.p and components.use_arma_errors.
    ma_coefs: array-like of floats or None
        MA(q) parameters used for residuals modeling.
        Empty array when there is no ARMA residuals modelling.
        Also see components.q and components.use_arma_errors.
    x0: array-like of floats or None
        Seed state for computations consisting of trend, seasonal and ARMA related seeds.
    """

    def __init__(self, components, alpha, beta=None, phi=None, box_cox_lambda=None,
                 gamma_params=None,
                 ar_coefs=None, ma_coefs=None, x0=None):
        """Holds all parameters needed to calculate model and forecasts

        Parameters
        ----------
        components: Components
            Defines which components are used in the model (trend, box-cox, seasonalities, ...)
        alpha: float
            Smoothing parameter value
        beta: float or None
            Trend smoothing parameter value.
            None when trend is not used.
            Also see components.use_trend.
        phi: float or None
            Trend damping parameter value.
            None when damping is not used.
            Also see components.use_damped_trend.
        box_cox_lambda: float or None
            Box-Cox transformation lambda parameter value
            None when series is not being transformed.
            Also see components.use_box_cox.
        gamma_params: array-like of floats or None
            Seasonal smoothing parameters.
            Also see components.seasonal_periods
        ar_coefs: array-like of floats or None
            AR(p) parameters used for residuals modeling.
            Also see components.p and components.use_arma_errors.
        ma_coefs: array-like of floats or None
            MA(q) parameters used for residuals modeling.
            Also see components.q and components.use_arma_errors.
        x0: array-like of floats or None
            Seed state for computations consisting of trend, seasonal and ARMA related seeds.
            When not provided will be initialized to a vector of zeroes.
        """
        self.components = components
        self.alpha = alpha
        self.box_cox_lambda = None
        if self.components.use_box_cox:
            self.box_cox_lambda = box_cox_lambda
        self.beta = None
        self.phi = None
        if self.components.use_trend:
            self.beta = beta
            self.phi = 1.
            if self.components.use_damped_trend:
                self.phi = phi
        self.gamma_params = self._normalize_gamma_params(gamma_params)

        self.__init_arma(ar_coefs, ma_coefs)

        self.__init_x0(x0)

    @classmethod
    def with_default_starting_params(cls, y, components):
        # abstract method
        raise NotImplementedError()

    def seasonal_components_amount(self):
        # abstract method
        raise NotImplementedError()

    def _normalize_gamma_params(self, gamma_params=None):
        # abstract method
        raise NotImplementedError()

    @classmethod
    def find_initial_box_cox_lambda(cls, y, components):
        """ Chooses starting Box-Cox lambda parameter using Guerrero method.

        Parameters
        ----------
        y: array-like
            Time series
        components: Components
            Components of the model

        Returns
        -------
        float or None
            Lambda parameter or None when no Box-Cox transformation applies.
        """
        if not components.use_box_cox:
            return None
        return transformation.find_box_cox_lambda(
            y,
            seasonal_periods=components.seasonal_periods,
            bounds=components.box_cox_bounds,
        )

    def with_arma(self, p=0, q=0):
        """Returns copy of itself with provided ARMA levels.

        ARMA coefficients are initialized to vectors of lengths p and q of zero values.

        Parameters
        ----------
        p: int
            Auto-regressive level p
        q: int
            Moving average level q

        Returns
        -------
        ModelParams
            Copy with ARMA of provided levels
        """
        me = copy.deepcopy(self)
        me.components = self.components.with_arma(p, q)
        me.__init_arma()
        return me

    def with_zero_x0(self):
        """Returns a copy of itself with seed state values set to zeroes.

        Returns
        -------
        ModelParams
            Copy with x0 as a vector of zeroes
        """
        me = copy.deepcopy(self)
        me.__init_x0()
        return me

    def with_x0(self, x0):
        """Returns a copy of itself with seed states set to provided values.

        Parameters
        ----------
        x0: array-like of floats
            Seed states

        Returns
        -------
        ModelParams
            Copy with x0 set to provided values
        """
        me = copy.deepcopy(self)
        me.__init_x0(x0)
        return me

    def with_vector_values(self, vector):
        """Returns a copy of itself with model parameters taken from the vector

        If necessary x0 is re-transformed to match new Box-Cox lambda parameter.

        Parameters
        ----------
        vector: array-like of floats
            A vector with model parameters. When all parameters are used it consist of:
            (alpha, box_cox_lambda, beta, phi, gamma parameters, AR coefficients, MA coefficients)
            When some components are not used they are also not present in the vector.

        Returns
        -------
        ModelParams
            A copy of itself with model parameters taken from the vector
        """
        x0 = self.x0.copy()

        alpha = vector[0]

        offset = 1
        boxcox = None
        if self.components.use_box_cox:
            boxcox = vector[offset]
            # recalculate x0 according to changed box-cox parameter
            x0 = transformation.boxcox(
                transformation.inv_boxcox(x0, self.box_cox_lambda),
                boxcox
            )

            offset += 1

        beta = None
        if self.components.use_trend:
            beta = vector[offset]
            offset += 1

        phi = None
        if self.components.use_damped_trend:  # note that damped trend is used only when trend is used
            phi = vector[offset]
            offset += 1

        gamma_params = vector[offset:(offset + self.components.gamma_params_amount())]  # may be empty
        offset += len(self.components.seasonal_periods)

        ar_coefs = vector[offset:(offset + self.components.p)]  # may be empty
        offset += self.components.p

        ma_coefs = vector[offset:(offset + self.components.q)]  # may be empty
        # offset += self.components.q

        return self.__class__(components=self.components,
                              alpha=alpha, beta=beta, phi=phi, box_cox_lambda=boxcox,
                              gamma_params=gamma_params,
                              ar_coefs=ar_coefs, ma_coefs=ma_coefs, x0=x0)

    def _create_x0_of_zeroes(self):
        """Creates seed vector of proper length of zeroes"""
        x0_length = 1
        if self.components.use_trend:
            x0_length += 1

        x0_length += self.seasonal_components_amount()

        x0_length += self.components.arma_length()

        return np.asarray([0] * x0_length)

    def to_vector(self):
        v = [[self.alpha]]
        if self.components.use_box_cox:
            v.append([self.box_cox_lambda])
        if self.components.use_trend:
            v.append([self.beta])
        if self.components.use_damped_trend:
            v.append([self.phi])
        if self.components.gamma_params_amount() > 0:
            v.append(self.gamma_params)
        if self.components.p > 0:
            v.append(self.ar_coefs)
        if self.components.q > 0:
            v.append(self.ma_coefs)
        return np.concatenate(v)

    def amount(self):
        """
        Amount of parameters in the model, including seed state
        :return: int
        """
        amount = 1
        if self.components.use_trend:
            amount += 1
        if self.components.use_damped_trend:
            amount += 1
        if self.components.use_box_cox:
            amount += 1

        return len(self.x0) + amount + self.components.gamma_params_amount() + self.components.arma_length()

    def is_box_cox_in_bounds(self):
        return self.components.is_box_cox_in_bounds(self.box_cox_lambda)

    def summary(self):
        s = self.components.summary()
        if not self.box_cox_lambda is None:
            s += 'Box-Cox Lambda %f\n' % self.box_cox_lambda
        s += 'Smoothing (Alpha): %f\n' % self.alpha
        if not self.beta is None:
            s += 'Trend (Beta): %f\n' % self.beta
        if not self.phi is None:
            s += 'Damping Parameter (Phi): %f\n' % self.phi
        s += 'Seasonal Parameters (Gamma): %s\n' % self.gamma_params
        s += 'AR coefficients %s\n' % self.ar_coefs
        s += 'MA coefficients %s\n' % self.ma_coefs
        s += 'Seed vector %s\n' % self.x0
        return s

    def __init_arma(self, ar_coefs=None, ma_coefs=None):
        self.ar_coefs = ArrayHelper.to_array(ar_coefs)
        self.ma_coefs = ArrayHelper.to_array(ma_coefs)

        if self.components.p > 0 and len(self.ar_coefs) == 0:
            self.ar_coefs = np.asarray([0] * self.components.p)

        if self.components.q > 0 and len(self.ma_coefs) == 0:
            self.ma_coefs = np.asarray([0] * self.components.q)

    def __init_x0(self, x0=None):
        self.x0 = x0
        if self.x0 is None:
            self.x0 = self._create_x0_of_zeroes()
