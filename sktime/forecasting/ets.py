__author__ = ["Hongyi Yang"]
__all__ = ["AutoETS"]

import numpy as np
import pandas as pd
from sktime.forecasting.base._sktime import BaseSktimeForecaster
from sktime.utils.boxcox import boxcox

class AutoETS(BaseSktimeForecaster):
    """
    Exponential smoothing state space model ported from R implementation.

    The methodology is fully automatic. The model is chosen automatically
    if not specified.

    Parameters
    ----------
    model : str, optional (default='ZZZ')
        A three-character string identifying method using the framework
        terminology of Hyndman et al. (2002) and Hyndman et al. (2008).
        The fisrt letter denotes the error type ("A", "M" or "Z");
        the second letter denotes the trend type ("N", "A", "M" or "Z");
        and the third letter denotes the season type ("N", "A", "M" or "Z").
        In all cases, "N" = none, "A" = additive, "M" = multiplicative
        and "Z" = automatically selected.
        It is also possible for the model to be of class AutoETS, and equal to
        the output from a previous call to AutoETS. In this case, the same
        model is fitted without re-estimating any smoothing parameters. See
        also the use_initial_values parameter.
    damped : bool, optional (default=)
        If True, use a damped trend (either additive or multiplicative). If
        False, both damped and non-damped trends will be tried and the best
        model (according to the information criterion) is returned.
    alpha: float optional (default=None)
        Value of alpha. If None, it is estimated.
    beta: float optional (default=None)
        Value of beta. If None, it is estimated.
    gamma: float optional (default=None)
        Value of gamma. If None, it is estimated.
    phi: float optional (default=None)
        Value of phi. If None, it is estimated.
    additive_only : bool, optional (default=False)
        If True, only additive models will be considered.
    lmbda: str, optional (default=None)
        Box-Cox transformation parameter. If 'auto', a transformation
        is automatically selected using boxcox.lmbda. If None, the
        transformation is ignored. Otherwise, data transformed before
        model is estimated. When lmbda is specified, additive_only is set
        to True.
    lower: optional (default=[0.0001,0.0001,0.0001,0.8])
        Lower bounds for the parameters [alpha, beta, gamma, phi].
    upper: optional (default=[0.9999,0.9999,0.9999,0.98)
        Upper bounds for the parameters [alpha, beta, gamma, phi].
    optimisation_criterion: str, optional (default='lik')
        Optimisation criterion, can be chosen from among the following strings:

        'mse' for Mean Square Error
        'amse' for Average MSE over first nmse forecast horizons
        'sigma' for Standard deviation of residuals)
        'mae' for Mean of absolute residuals
        'lik' for Log-likelihood
    nmse: int, optional
        Number of steps for average multistep MSE (1<=nmse<=30).
    bounds: str, optional (default='both')
        Type of parameter space to impose:

        'usual' for all parameters must lie between specified lower and upper
            bounds
        'admissible' for parameters must lie in the admissible space
        'both' for parameters take intersection of these regions
    information_criterion : str optional (default=)
        The information criterion used to select the best ETS model.
    restrict: bool, optional (default=True)
        If True, the models with infinite variance will not be allowed.
    allow_multiplicative_trend: bool optional (default=)
        If True, models with multiplicative trend are allowed when searching
        for a model. Otherwise, the model space excludes them. This argument is
        ignored if a multiplicative trend model is explicitly requested (e.g.,
        using model = 'MMN').
    use_initial_values: bool, optioanl (default=False)
        If True and model is of class AutoETS, then the initial values in the
        model are also not re-estimated.
    """

    def __init__(self, model='ZZZ', damped='False', alpha=None, beta=None,
                 gamma=None, phi=None, additive_only=False, lmbda=None,
                 lower=[0.0001, 0.0001, 0.0001, 0.8],
                 upper=[0.9999, 0.9999, 0.9999, 0.98],
                 optimisation_criterion='lik', nmse=, bounds='both',
                 information_criterion=, restrict=True,
                 allow_multiplicative_trend=, use_initial_values=False):

        self.model = model
        self.damped = damped
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.additive_only = additive_only
        self.lmbda = lmbda
        self.lower = lower
        self.upper = upper
        self.optimisation_criterion = optimisation_criterion
        self.nmse = nmse
        self.bounds = bounds
        self.information_criterion = information_criterion
        self.restrict = restrict
        self.allow_multiplicative_trend = allow_multiplicative_trend
        self.use_initial_values = use_initial_values

        super(AutoETS, self).__init__()

    def fit(self, y_train):
        """ Fit to training data.



        """

        # Set lambda value
        # inherit from fitted model
        if isinstance(self.model, AutoETS) && self.lmbda is None:
            self.lmbda = self.model.lmbda

        # determine lmbda value
        if self.lmbda is None:
            y_train, self.lmbda = boxcox(y_train, self.lmbda)
            additive_only = True

        # Check for nmse value
        if nmse < 1 || nmse > 30:
            raise ValueError("nmse out of range (1 <= nmse <=30).")

        # Check for lower and upper limits
        if upper[0] < lower[0]:
            raise ValueError("Lower alpha limit must be less that upper limit")

        if upper[1] < lower[1]:
            raise ValueError("Lower beta limit must be less that upper limit")

        if upper[2] < lower[2]:
            raise ValueError("Lower gamma limit must be less that upper limit"

        if upper[3] < lower[3]:
            raise ValueError("Lower phi limit must be less that upper limit")

        # Refit model to new data if model is an AutoETS object
        if isinstance(self.model, AutoETS):
            # prevent alpha being 0
            self.alpha = max(self.model.alpha, 1e-10)
            self.beta = self.model.beta
            self.gamma = self.model.gamma
            self.phi = self.model.phi

    def _pegelsresid(y, m, init_state, errortype, trendtype, seasontype,
                     damped, alpha, beta, gamma, phi, nmse):
        n = len(y)
        p = len(init_state)
        x = np.zeros(p * (n + 1))
        x[0: p] = init_state
        e = np.zeros(n)
        lik = 0
        if not damped:
            phi = 1
        if trendtype == "N":
            beta = 0
        if seasontype == "N":
            gamma = 0

        amse = np.zeros(nmse)

        
