import numpy as np
from scipy.optimize import minimize

from .. import error


class ParamsOptimizer(object):
    """ Tries to find optimal seed and parameter values for the model

    Uses linear regression to choose seed values
    and Nelder-Mead optimization to choose model parameters

    Attributes
    ----------
    optimal_params: ModelParams or None
        Optimal model parameters that were found.
        None when no optimization was performed.

    Methods
    -------
    optimize(y, initial_params, calculate_seed=True)
        Seeks for the best seed and parameter values for the model defined by initial params
    converged()
        Returns true when optimization did converge
    optimal_model()
        Returns the best model that was found
    """

    def __init__(self, context):
        self.context = context

        # Initialize optimization params
        self.optimal_params = None  # no parameters found yet
        self._success = False  # did not converge yet
        self._y = None
        self._starting_params = None
        self._scale = None

    def converged(self):
        """Returns true when optimization converged

        Returns
        -------
        bool
            True when converged
        """
        return self._success

    def optimal_model(self):
        """Constructs the best model that was found

        Returns
        -------
        Model
            Optimal model
        """
        if self._y is None:
            self.context.get_exception_handler().exception("No optimization was performed yet", error.BatsException)
        # will return a model even if optimization did not converge
        model = self.context.create_model(self.optimal_params).fit(self._y)
        if not model.is_fitted:
            model.add_warning("Model did not calculate properly and seems unusable.")
        if not model.can_be_admissible():
            model.add_warning("Model is not admissible! Forecasts may be unstable. Check long term forecasts.")
        if not self.converged():
            model.add_warning("Optimization did not converge. Forecasts may be unstable.")
        return model

    def optimize(self, y, initial_params, calculate_seed=True):
        """Finds seed and parameters that minimize likelihood

        Parameters
        ----------
        y: array-like of floats
            Time series
        initial_params: ModelParams
            Model components and initial parameter values
        calculate_seed: bool, optional (default=True)
            If optimizer should also calculate seed x0.
            When False optimization shall use x0 from initial params as seed

        Returns
        -------
        ParamsOptimizer
            Itself
        """
        self._y = y
        self._scale = None  # reset scale

        # Initialize seed using linear regression
        self._starting_params = initial_params
        if calculate_seed:
            x0 = self._calculate_seed_x0(y, initial_params)
            self._starting_params = initial_params.with_x0(x0)

        starting_vector = self._starting_params.to_vector()
        result = minimize(
            self._scale_and_calculate_likelihood,  # function being optimized
            x0=self._inv_scale_vector(starting_vector),
            # self.calculate_likelihood,
            # x0=starting_vector,
            method='Nelder-Mead',
            options={
                'maxiter': 100 * (len(starting_vector) ** 2),
                'fatol': 1e-8,
                # 'disp': True,
            }
        )
        self._success = result.success

        self.optimal_params = self._starting_params.with_vector_values(
            self._scale_vector(result.x)
        )

        return self

    def _scale_and_calculate_likelihood(self, optimization_vector):
        """Scales parameters vector (for performance) and calculates model likelihood"""
        optimization_vector = self._scale_vector(optimization_vector)
        return self._calculate_likelihood(optimization_vector)

    def _calculate_likelihood(self, optimization_vector):
        """Fits model with provided parameters vector to time series and returns its likelihood"""
        infinity = 10 ** 10  # we can not return np.inf as optimization will not work
        # print(optimization_vector)
        params = self._starting_params.with_vector_values(optimization_vector)
        model = self.context.create_model(params, validate_input=False)
        if not model.can_be_admissible():  # don't even calculate the model for such params
            return infinity
        model = model.fit(self._y)
        likelihood = model.likelihood()
        if likelihood == np.inf:
            return infinity
        return likelihood

    def _calculate_seed_x0(self, y, params):
        """Calculates seed values using seed finder (linear regression)"""
        model = self.context.create_model(params.with_zero_x0(), validate_input=False)
        y_tilda = model.fit(y).resid_boxcox

        w = model.matrix.make_w_vector()

        D = model.matrix.calculate_D_matrix()

        w_tilda = np.zeros((len(y), len(w)))
        w_tilda[0, :] = w

        D_transposed = D.T
        for t in range(1, len(y)):
            w_tilda[t, :] = D_transposed @ w_tilda[t - 1, :]

        # TODO I am wondering, since we are removing this ARMA part, do we need to calculate it

        # linear regression to find x0
        seed_finder = self.context.create_seed_finder(params.components)
        return seed_finder.find(w_tilda, y_tilda)

    def _scale_vector(self, vector):
        """Scales vector"""
        scale = self._get_scale()
        return vector * scale

    def _inv_scale_vector(self, vector):
        """Scales vector back"""
        scale = self._get_scale()
        return vector / scale

    def _get_scale(self):
        """Provides vector value scaling factors used to speed-up optimization"""
        if self._scale is not None:
            return self._scale
        components = self._starting_params.components
        scale = [[0.01]]  # alpha scale
        if components.use_box_cox:
            scale.append([0.001])  # box-cox lambda scale
        if components.use_trend:
            scale.append([0.01])  # beta scale
            if components.use_damped_trend:
                scale.append([0.01])  # phi scale
        if components.gamma_params_amount() > 0:
            scale.append([1e-5] * components.gamma_params_amount())  # seasonal parameters scale
        if components.arma_length():
            scale.append([0.1] * components.arma_length())  # ARMA parameters scale
        self._scale = np.concatenate(scale)
        return self._scale
