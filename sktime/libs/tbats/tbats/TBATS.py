import numpy as np

from ..abstract import Estimator
from . import Context


class TBATS(Estimator):
    """
    TBATS estimator used to fit and select best performing model.

    TBATS (Exponential smoothing state space model with Box-Cox
    transformation, ARMA errors, Trigonometric Trend and Seasonal components.)

    Model has been described in De Livera, Hyndman & Snyder (2011).

    All of the useful methods have been implemented in parent Estimator class.
    """

    def __init__(self,
                 use_box_cox=None, box_cox_bounds=(0, 1),
                 use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 show_warnings=True,
                 n_jobs=None, multiprocessing_start_method='spawn',
                 context=None):
        """ Class constructor

        Parameters
        ----------
        use_box_cox: bool or None, optional (default=None)
            If Box-Cox transformation of original series should be applied.
            When None both cases shall be considered and better is selected by AIC.
        box_cox_bounds: tuple, shape=(2,), optional (default=(0, 1))
            Minimal and maximal Box-Cox parameter values.
        use_trend: bool or None, optional (default=None)
            Indicates whether to include a trend or not.
            When None both cases shall be considered and better is selected by AIC.
        use_damped_trend: bool or None, optional (default=None)
            Indicates whether to include a damping parameter in the trend or not.
            Applies only when trend is used.
            When None both cases shall be considered and better is selected by AIC.
        seasonal_periods: iterable or array-like of floats, optional (default=None)
            Length of each of the periods (amount of observations in each period).
            TBATS accepts int and float values here.
            When None or empty array, non-seasonal model shall be fitted.
        use_arma_errors: bool, optional (default=True)
            When True BATS will try to improve the model by modelling residuals with ARMA.
            Best model will be selected by AIC.
            If False, ARMA residuals modeling will not be considered.
        show_warnings: bool, optional (default=True)
            If warnings should be shown or not.
            Also see Model.warnings variable that contains all model related warnings.
        n_jobs: int, optional (default=None)
            How many jobs to run in parallel when fitting BATS model.
            When not provided BATS shall try to utilize all available cpu cores.
        multiprocessing_start_method: str, optional (default='spawn')
            How threads should be started.
            See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        context: abstract.ContextInterface, optional (default=None)
            For advanced users only. Provide this to override default behaviors
        """
        if context is None:
            # the default TBATS context
            context = Context(show_warnings, n_jobs=n_jobs, multiprocessing_start_method=multiprocessing_start_method)
        super().__init__(context, use_box_cox=use_box_cox, box_cox_bounds=box_cox_bounds,
                         use_trend=use_trend, use_damped_trend=use_damped_trend,
                         seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors,
                         n_jobs=n_jobs)

    def _normalize_seasonal_periods(self, seasonal_periods):
        return self._normalize_seasonal_periods_to_type(seasonal_periods, dtype=float)

    def _do_fit(self, y):
        """Checks various model combinations to find best one by AIC"""
        components_grid = self._prepare_non_seasonal_components_grid()
        non_seasonal_model = self._choose_model_from_possible_component_settings(y, components_grid=components_grid)
        harmonics_choosing_strategy = self.context.create_harmonics_choosing_strategy()
        chosen_harmonics = harmonics_choosing_strategy.choose(y, self.create_most_complex_components())
        components_grid = self._prepare_components_grid(seasonal_harmonics=chosen_harmonics)
        seasonal_model = self._choose_model_from_possible_component_settings(y, components_grid=components_grid)

        if non_seasonal_model.aic < seasonal_model.aic:
            return non_seasonal_model

        return seasonal_model

    def create_most_complex_components(self):
        """Creates model components for the most complex model without ARMA residuals modelling"""
        components = dict(
            use_box_cox=self.use_box_cox is not False,
            use_trend=self.use_trend is not False,
            use_damped_trend=self.use_damped_trend is not False,
            seasonal_periods=self.seasonal_periods,
            use_arma_errors=False,
        )
        return self.context.create_components(**components)
