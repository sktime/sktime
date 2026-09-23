import numpy as np
import multiprocessing as actual_processing
import multiprocessing.dummy as dummy_processing
from sklearn.base import BaseEstimator
from sklearn.utils.validation import column_or_1d as c1d

from .._compat import check_array
from sklearn.model_selection import ParameterGrid

from .. import error


class Estimator(BaseEstimator):
    """Base estimator for BATS and TBATS models

    Methods
    -------
    fit(y)
        Fit to y and select best performing model based on AIC criterion.
    """

    def __init__(self, context,
                 use_box_cox=None, box_cox_bounds=(0, 1),
                 use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 n_jobs=None):
        """ Class constructor

        Parameters
        ----------
        context: abstract.ContextInterface
            For advanced users only. Provide this to override default behaviors
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
        seasonal_periods: iterable or array-like, optional (default=None)
            Length of each of the periods (amount of observations in each period).
            BATS accepts only int values here.
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
        """
        self.context = context
        self.n_jobs = n_jobs

        self.seasonal_periods = self._normalize_seasonal_periods(seasonal_periods)
        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_arma_errors = use_arma_errors
        self.use_trend = use_trend
        if use_trend is False:
            if use_damped_trend is True:
                self.context.get_exception_handler().warn(
                    "When use_damped_trend can be used only with use_trend. Setting damped trend to False.",
                    error.InputArgsWarning
                )
            use_damped_trend = False
        self.use_damped_trend = use_damped_trend

    def _normalize_seasonal_periods(self, seasonal_periods):
        # abstract method
        raise NotImplementedError()

    def _do_fit(self, y):
        # abstract method
        raise NotImplementedError()

    def fit(self, y):
        """Fit model to observations ``y``.

        :param y: array-like or iterable, shape=(n_samples,)
        :return: abstract.Model, Fitted model
        """
        y = self._validate(y)
        if y is False:
            # Input data is not valid and no exception was raised yet.
            # This can happen only when one overrides default exception handler (see tbats.error.ExceptionHandler)
            return None

        if np.allclose(y, y[0]):
            return self.context.create_constant_model(y[0]).fit(y)

        best_model = self._do_fit(y)

        for warning in best_model.warnings:
            self.context.get_exception_handler().warn(warning, error.ModelWarning)

        return best_model

    def _validate(self, y):
        """Validates input time series. Also adjusts box_cox if necessary."""
        try:
            y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                                copy=True, dtype=np.float64))  # type: np.ndarray
        except Exception as validation_exception:
            self.context.get_exception_handler().exception(
                "y series is invalid", error.InputArgsException, previous_exception=validation_exception
            )
            return False

        if np.any(y <= 0):
            if self.use_box_cox is True:
                self.context.get_exception_handler().warn(
                    "Box-Cox transformation (use_box_cox) was forced to True "
                    "but there are negative values in input series. "
                    "Setting use_box_cox to False.",
                    error.InputArgsWarning
                )
            self.use_box_cox = False

        return y

    def _case_fit(self, components_combination):
        """Internal method used by parallel computation."""
        case = self.context.create_case_from_dictionary(**components_combination)
        return case.fit(self._y)

    def _choose_model_from_possible_component_settings(self, y, components_grid):
        """Fits all models in a grid and returns best one by AIC

        Returns
        -------
            abstract.Model
                Best model by AIC
        """
        self._y = y
        # note n_jobs = None means to use cpu_count()
        with self.context.multiprocessing().Pool(processes=self.n_jobs) as pool:
            models = pool.map(self._case_fit, components_grid)
            pool.close()
        self._y = None  # clean-up
        if len(models) == 0:
            return None
        best_model = models[0]
        for model in models:
            if model.aic < best_model.aic:
                best_model = model
        return best_model

    def _prepare_components_grid(self, seasonal_harmonics=None):
        """Provides a grid of all allowed model component combinations.

        Parameters
        ----------
        seasonal_harmonics: array-like or None
            When provided all component combinations shall contain those harmonics
        """
        allowed_combinations = []

        use_box_cox = self.use_box_cox

        base_combination = {
            'use_box_cox': self.__prepare_component_boolean_combinations(use_box_cox),
            'box_cox_bounds': [self.box_cox_bounds],
            'use_arma_errors': [self.use_arma_errors],
            'seasonal_periods': [self.seasonal_periods],
        }
        if seasonal_harmonics is not None:
            base_combination['seasonal_harmonics'] = [seasonal_harmonics]

        if self.use_trend is not True:  # False or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [False],
                    'use_damped_trend': [False],  # Damped trend must be False when trend is False
                }
            })

        if self.use_trend is not False:  # True or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [True],
                    'use_damped_trend': self.__prepare_component_boolean_combinations(self.use_damped_trend),
                }
            })
        return ParameterGrid(allowed_combinations)

    def _prepare_non_seasonal_components_grid(self):
        """Provides a grid of all allowed  non-season model component combinations."""
        allowed_combinations = []

        use_box_cox = self.use_box_cox

        base_combination = {
            'use_box_cox': self.__prepare_component_boolean_combinations(use_box_cox),
            'box_cox_bounds': [self.box_cox_bounds],
            'use_arma_errors': [self.use_arma_errors],
            'seasonal_periods': [[]],
        }

        if self.use_trend is not True:  # False or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [False],
                    'use_damped_trend': [False],  # Damped trend must be False when trend is False
                }
            })

        if self.use_trend is not False:  # True or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [True],
                    'use_damped_trend': self.__prepare_component_boolean_combinations(self.use_damped_trend),
                }
            })
        return ParameterGrid(allowed_combinations)

    @staticmethod
    def __prepare_component_boolean_combinations(param):
        combinations = [param]
        if param is None:
            combinations = [False, True]
        return combinations

    def _normalize_seasonal_periods_to_type(self, seasonal_periods, dtype):
        """Validates seasonal periods and normalizes them

        Normalization ensures periods are of proper type, unique and sorted.
        """
        if seasonal_periods is not None:
            try:
                seasonal_periods = c1d(check_array(seasonal_periods, ensure_2d=False, force_all_finite=True,
                                                   ensure_min_samples=0,
                                                   copy=True, dtype=dtype))
            except Exception as validation_exception:
                self.context.get_exception_handler().exception("seasonal_periods definition is invalid",
                                                               error.InputArgsException,
                                                               previous_exception=validation_exception)

            seasonal_periods = np.unique(seasonal_periods)
            if len(seasonal_periods[np.where(seasonal_periods <= 1)]) > 0:
                self.context.get_exception_handler().warn(
                    "All seasonal periods should be values greater than 1. "
                    "Ignoring all seasonal period values that do not meet this condition.",
                    error.InputArgsWarning
                )
            seasonal_periods = seasonal_periods[np.where(seasonal_periods > 1)]
            seasonal_periods.sort()
            if len(seasonal_periods) == 0:
                seasonal_periods = None
        return seasonal_periods
