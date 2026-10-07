import copy
import numpy as np


class Components(object):
    """Contains all the information necessary to determine the amount and the structure of parameters

    Parameters
    ----------
    use_box_cox: bool
        If box cox will be used
    box_cox_bounds: tuple, shape=(2,)
        Minimal and maximal allowed values for Box-Cox lambda transformation.
    use_trend: bool
        If trend will be used
    use_damped_trend: bool
        If trend damping will apply. Applicable only when use_trend=True
    seasonal_periods: array-like
        Seasonal period lengths (amount of observations for one period of each seasonality)
    use_arma_errors:
        If ARMA(p,q) residuals modelling shall be tried.
    p: int
        Amount of AR(p) terms for ARMA(p,q) residuals modelling.
        If use_arma_errors=False then p=0
    q: int
        Amount of MA(p) terms for ARMA(p,q) residuals modelling.
        If use_arma_errors=False then q=0
    """

    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        """Contains all the information necessary to determine the amount and the structure of parameters

        Does not contain model parameter values

        Parameters
        ----------
        use_box_cox: bool, optional (default=False)
            If box cox will be used
        use_trend: bool, optional (default=False)
            If trend will be used
        use_damped_trend: bool, optional (default=False)
            If trend damping will apply. Applicable only when use_trend=True
        seasonal_periods: array-like or None, optional (default=None)
            Seasonal period lengths (amount of observations for one period of each seasonality)
            When None, there there will be no seasonal periods
        use_arma_errors: bool, optional (default=False)
            If ARMA(p,q) residuals modelling shall be tried.
        p: int, optional (default=0)
            Amount of AR(p) terms for ARMA(p,q) residuals modelling.
            If use_arma_errors=False then p=0
        q: int, optional (default=0)
            Amount of MA(p) terms for ARMA(p,q) residuals modelling.
            If use_arma_errors=False then q=0
        box_cox_bounds: tuple, shape=(2,), optional (default=(0, 1))
            Minimal and maximal allowed values for Box-Cox lambda transformation.
        """
        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = False
        if self.use_trend:  # only when trend is used
            self.use_damped_trend = use_damped_trend
        self.seasonal_periods = self._normalize_seasons(seasonal_periods)
        self.__use_arma(do_use=use_arma_errors, p=p, q=q)

    @classmethod
    def create_constant_components(cls):
        # abstract method
        raise NotImplementedError()

    def _normalize_seasons(self, seasonal_periods):
        # abstract method
        raise NotImplementedError()

    def seasonal_components_amount(self):
        # abstract method
        raise NotImplementedError()

    def gamma_params_amount(self):
        # abstract method
        raise NotImplementedError()

    def _seasonal_summary(self):
        # abstract method
        raise NotImplementedError()

    def arma_length(self):
        """Returns amount of ARMA(p,q) parameters"""
        return self.p + self.q

    def summary(self):
        """Returns components summary

        Returns
        -------
        str
            summary of components
        """
        s = "Use Box-Cox: %r\n" % self.use_box_cox
        s += "Use trend: %r\n" % self.use_trend
        s += "Use damped trend: %r\n" % self.use_damped_trend
        s += "Seasonal periods: %s\n" % self.seasonal_periods
        s += self._seasonal_summary()
        s += "ARMA errors (p, q): (%d, %d)\n" % (self.p, self.q)
        return s

    def with_seasonal_periods(self, seasonal_periods):
        """Creates copy of itself but with new seasonal periods

        Parameters
        ----------
        seasonal_periods: array-like
            New season lengths

        Returns
        -------
        Components
            copy of components with new seasonal periods
        """
        components = copy.deepcopy(self)
        components.seasonal_periods = self._normalize_seasons(seasonal_periods)
        return components

    def without_seasonal_periods(self):
        """Creates copy itself without seasonality

        Returns
        -------
        Components
            copy of components without seasonal periods
        """
        components = copy.deepcopy(self)
        components.seasonal_periods = np.asarray([])
        return components

    def without_arma(self):
        """Creates copy itself without ARMA components

        Returns
        -------
        Components
            copy of components without ARMA
        """
        components = copy.deepcopy(self)
        return components.__use_arma(False)

    def with_arma(self, p=0, q=0):
        """Creates copy itself with provided ARMA components

        Returns
        -------
        Components
            copy of components with provided ARMA(p,q) degrees
        """
        components = copy.deepcopy(self)
        return components.__use_arma(do_use=True, p=p, q=q)

    def is_box_cox_in_bounds(self, box_cox_lambda):
        """Tells if provided Box-Cox lambda is within allowed interval

        When box cox is not used, returns True regardless of lambda value.

        Returns
        -------
        bool
            True when lambda is in bounds
        """
        if self.use_box_cox:
            return self.box_cox_bounds[0] <= box_cox_lambda <= self.box_cox_bounds[1]
        return True

    def __use_arma(self, do_use=True, p=0, q=0):
        """Modifies object state so that ARMA parameters according to provided"""
        self.use_arma_errors = do_use
        self.p = 0
        self.q = 0
        if do_use:  # only when ARMA errors are used
            self.p = p
            self.q = q
        return self
