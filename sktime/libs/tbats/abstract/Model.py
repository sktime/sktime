import numpy as np
from sklearn.utils.validation import column_or_1d as c1d

from .._compat import check_array
import scipy.stats as stats
from numpy.linalg import LinAlgError
import warnings

from .. import error, transformation


class Model(object):
    """BATS or TBATS model

    Attributes
    ----------
    warnings: array-like
        All warning messages associated to the model.
        Empty array when there are no warnings.
    is_fitted: bool
        If the model has been successfully fitted
        False when model was not fitted at all or fitting of the model failed
    y: array-like or None
        Time series model has been fitted to.
        None when no fitting was performed yet.
    y_hat: array-like or None
        In series predictions.
    resid: array-like or None
        Residuals from in series predictions
    resid_boxcox: array-like or None
        Residuals from Box-Cox transformed predictions.
        When no box cox is used those are equal to resid.
    x_last: array-like or None
        last vector from x matrix that can be used to calculate predictions
    aic: float
        AIC criterion value
        np.inf when not fitted or failed to fit
    params: ModelParams
        Parameters used in the model
    matrix: MatrixBuilderInterface
        Matrices used in linear equations of the model

    Methods
    -------
    fit(y)
        Calculates in-series predictions for provided time series.
        Can also be used to re-fit the model.
    forecast(steps=1)
        Calculates forecast for provided amount of steps ahead.
    """

    def __init__(self, model_params, context, validate_input=True):
        """Prepares model for fitting.

        Do not use this constructor directly. See abstract.Context methods.

        Parameters
        ----------
        model_params: abstract.ModelParams
            All model parameters and components needed for calculation.
        context: abstract.ContextInterface
            The context used for implementation details.
        validate_input: bool
            If input time series y should be trusted to be valid or not.
            Used for performance.
        """
        self.context = context
        self.warnings = []
        self.params = model_params
        self.validate_input = validate_input
        self.matrix = self.context.create_matrix_builder(self.params)

        self.is_fitted = False
        self.y = None
        self.y_hat = None
        self.resid_boxcox = None
        self.resid = None
        self.x_last = None
        self.aic = np.inf

    def fit(self, y):
        """Calculates in series predictions for provided time series.

        Parameters
        ----------
        y: array-like
            Time series to fit model to

        Returns
        -------
        self
            See class attributes to read fit results
        """
        return self._fit_to_observations(y, self.params.x0)

    def forecast(self, steps=1, confidence_level=None):
        """Forecast for provided amount of steps ahead

        When confidence_level it will also return confidence bounds. Their calculation is valid
        under assumption that residuals are from normal distribution.

        Parameters
        ----------
        steps: int
            Amount of steps to forecast
        confidence_level: float, optional (default=None)
            When provided and a value between 0 and 1 also confidence bounds shall be returned for provided level.
            If None (default), confidence intervals will not be calculated.

        Returns
        -------
        array-like:
            Forecasts
        dict:
            Confidence bounds, present only when confidence_level is provided
        """
        if not self.is_fitted:
            self.context.get_exception_handler().exception(
                'Model must be fitted to be able to forecast. Use fit method first.',
                error.BatsException
            )
        steps = int(steps)
        if steps < 1:
            self.context.get_exception_handler().exception(
                'Parameter \'steps\' must be a positive integer',
                error.InputArgsException
            )

        F = self.matrix.make_F_matrix()
        w = self.matrix.make_w_vector()

        # initialize matrices
        yw_hat = np.asarray([0.0] * steps)
        x = self.x_last

        for t in range(0, steps):
            yw_hat[t] = w @ x
            x = F @ x

        y_hat = self._inv_boxcox(yw_hat)

        if confidence_level is None:
            return y_hat

        return y_hat, self._calculate_confidence_intervals(y_hat, confidence_level)

    def summary(self):
        """Returns model summary containing all parameter values.

        Returns
        -------
        str
            Model summary
        """
        str = ''
        str += self.params.summary() + '\n'
        str += 'AIC %f' % self.aic
        return str

    def likelihood(self):
        """Calculates likelihood of the model. Used for optimization."""
        if not self.is_fitted:
            return np.inf

        residuals = self.resid_boxcox

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                likelihood_part = len(residuals) * np.log(np.sum(residuals * residuals))
            except RuntimeWarning:
                # calculation issues, values close to max float value
                return np.inf

        boxcox_part = 0
        if self.params.components.use_box_cox:
            boxcox_part = 2 * (self.params.box_cox_lambda - 1) * np.sum(np.log(self.y))

        return likelihood_part - boxcox_part

    def calculate_aic(self):
        """Calculates AIC criterion value for the model"""
        likelihood = self.likelihood()
        if likelihood == np.inf:
            return np.inf
        return likelihood + 2 * self.params.amount()

    def _fit_to_observations(self, y, starting_x):
        """Fits model with starting x to time series"""
        self.warnings = []
        self.is_fitted = False

        if self.validate_input:
            try:
                y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                                    copy=True, dtype=np.float64))  # type: np.ndarray
            except Exception as validation_exception:
                self.context.get_exception_handler().exception("y series is invalid",
                                                               error.InputArgsException,
                                                               previous_exception=validation_exception)
        self.y = y
        yw = self._boxcox(y)

        matrix_builder = self.matrix
        w = matrix_builder.make_w_vector()
        g = matrix_builder.make_g_vector()
        F = matrix_builder.make_F_matrix()

        # initialize matrices
        yw_hat = np.asarray([0.0] * len(y))
        # x = np.matrix(np.zeros((len(params.x0), len(yw) + 1)))
        x = starting_x

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for t in range(0, len(y)):
                    yw_hat[t] = w @ x
                    e = yw[t] - yw_hat[t]
                    x = F @ x + g * e
            except RuntimeWarning:
                # calculation issues, values close to max float value
                self.add_warning('Numeric calculation issues detected. Model is not usable.')
                self.is_fitted = False
                return self

        # store fit results
        self.x_last = x
        self.resid_boxcox = yw - yw_hat
        try:
            self.y_hat = self._inv_boxcox(yw_hat)
        except RuntimeWarning:
            self.add_warning('Box-Cox related numeric calculation issues detected. Model is not usable.')
            self.is_fitted = False
            return self

        self.resid = self.y - self.y_hat

        self.is_fitted = True
        self.aic = self.calculate_aic()

        return self

    def _boxcox(self, y):
        yw = y
        if self.params.components.use_box_cox:
            yw = transformation.boxcox(y, lam=self.params.box_cox_lambda)
        return yw

    def _inv_boxcox(self, yw):
        y = yw
        if self.params.components.use_box_cox:
            y = transformation.inv_boxcox(yw, lam=self.params.box_cox_lambda, force_valid=True)
        return y

    def _calculate_confidence_intervals(self, predictions, level):
        """Calculates confidence intervals

        Parameters
        ----------
        predictions: array-like
            Predictions from the model
        level: float
            Confidence level as a number between 0 and 1.
            For example 0.95 denotes 95% confidence interval.

        Returns
        -------
        dict
            mean (predictions), lower_bound, upper_bound, std, calculated_for_level
        -------

        """
        F = self.matrix.make_F_matrix()
        g = self.matrix.make_g_vector()
        w = self.matrix.make_w_vector()

        c = np.asarray([1.0] * len(predictions))
        f_running = np.identity(F.shape[1])
        for step in range(1, len(predictions)):
            c[step] = w @ f_running @ g
            f_running = f_running @ F
        variance_multiplier = np.cumsum(c * c)

        base_variance_boxcox = np.sum(self.resid_boxcox * self.resid_boxcox) / len(self.y)
        variance_boxcox = base_variance_boxcox * variance_multiplier
        std_boxcox = np.sqrt(variance_boxcox)

        z_scores = std_boxcox * np.abs(stats.norm.ppf((1 - level) / 2))
        lower_bound_boxcox = self._boxcox(predictions) - z_scores
        upper_bound_boxcox = self._boxcox(predictions) + z_scores

        return dict(
            mean=predictions,
            lower_bound=self._inv_boxcox(lower_bound_boxcox),
            upper_bound=self._inv_boxcox(upper_bound_boxcox),
            calculated_for_level=level,
        )

    def can_be_admissible(self):
        """Tells if model can be admissible."""
        if not self.params.is_box_cox_in_bounds():
            return False

        params = self.params
        if params.components.use_damped_trend and (params.phi < 0.8 or params.phi > 1):
            return False

        if not self.__AR_is_stationary(params.ar_coefs):
            return False

        if not self.__MA_is_invertible(params.ma_coefs):
            return False

        D = self.matrix.calculate_D_matrix()
        return self.__D_matrix_eigen_values_check(D)

    def is_admissible(self):
        """Tells if model is admissible (stable). Model that has not been fitted is not addmisible."""
        if not self.is_fitted:
            return False
        return self.can_be_admissible()

    @staticmethod
    def __AR_is_stationary(ar_coefs):
        # cut out trailing and non-significant AR components
        significant_indices = np.where(np.abs(ar_coefs) > 1e-08)[0]
        if len(significant_indices) == 0:
            return True
        p = np.max(significant_indices) + 1
        significant_ar_coefs = ar_coefs[0:p]

        roots = np.polynomial.polynomial.polyroots(np.concatenate([[1], -significant_ar_coefs]))
        # Note that np.abs also works with complex numbers. It provides length of complex number
        # There should be no roots in the unit circle
        return np.all(np.abs(roots) > 1.0)

    @staticmethod
    def __MA_is_invertible(ma_coefs):
        # cut out trailing and non-significant AR components
        significant_indices = np.where(np.abs(ma_coefs) > 1e-08)[0]
        if len(significant_indices) == 0:
            return True
        q = np.max(significant_indices) + 1
        significant_ma_coefs = ma_coefs[0:q]

        roots = np.polynomial.polynomial.polyroots(np.concatenate([[1], significant_ma_coefs]))
        # Note that np.abs also works with complex numbers. It provides length of complex number
        # There should be no roots in the unit circle
        return np.all(np.abs(roots) > 1.0)

    @staticmethod
    def __D_matrix_eigen_values_check(D):
        try:
            eigen_values = np.linalg.eigvals(D)
        except LinAlgError:
            return False
        return np.all(np.abs(eigen_values) < 1.01)

    def add_warning(self, message):
        """Add a warning message to the model

        Parameters
        ----------
        message: str
            The message
        """
        self.warnings.append(message)
