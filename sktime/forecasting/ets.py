__author__ = ["Hongyi Yang"]
__all__ = ["AutoETS"]

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sktime.forecasting.base._sktime import BaseSktimeForecaster
from sktime.utils.boxcox import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose as decompose
from scipy.stats import linregress
from scipy.optimize import minimize


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
    sp: int, (default=1)
        Seasonal periodicity of the time series.
        Details of the definition can be found at:
        https://robjhyndman.com/hyndsight/seasonal-periods/
    damped : bool, optional (default=[True, False])
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
    lmbda: float or str, optional (default=None)
        Box-Cox transformation parameter. If 'auto', a transformation
        is automatically selected using boxcox.lmbda. If None, the
        transformation is ignored. Otherwise, data transformed before
        model is estimated. When lmbda is specified, additive_only is set
        to True.
    lower: optional (default=[0.0001,0.0001,0.0001,0.8])
        Lower bounds for the parameters [alpha, beta, gamma, phi].
    upper: optional (default=[0.9999,0.9999,0.9999,0.98)
        Upper bounds for the parameters [alpha, beta, gamma, phi].
    optimisation_criterion: str, optional (default=['lik', 'amse', 'mse',
                                                    'sigma', 'mae'])
        Optimisation criterion, can be chosen from among the following strings:

        'mse' for Mean Square Error
        'amse' for Average MSE over first nmse forecast horizons
        'sigma' for Standard deviation of residuals)
        'mae' for Mean of absolute residuals
        'lik' for Log-likelihood
    nmse: int, optional (default=3)
        Number of steps for average multistep MSE (1<=nmse<=30).
    bounds: str, optional (default=['both', 'usual', 'admissible'])
        Type of parameter space to impose:

        'usual' for all parameters must lie between specified lower and upper
            bounds
        'admissible' for parameters must lie in the admissible space
        'both' for parameters take intersection of these regions
    information_criterion : str optional (default=['aicc', 'aic', 'bic'])
        The information criterion used to select the best ETS model.
    restrict: bool, optional (default=True)
        If True, the models with infinite variance will not be allowed.
    allow_multiplicative_trend: bool optional (default=False)
        If True, models with multiplicative trend are allowed when searching
        for a model. Otherwise, the model space excludes them. This argument is
        ignored if a multiplicative trend model is explicitly requested (e.g.,
        using model = 'MMN').
    use_initial_values: bool, optioanl (default=False)
        If True and model is of class AutoETS, then the initial values in the
        model are also not re-estimated.
    """

    def __init__(self, model='ZZZ', sp=1, damped=['True', 'False'],
                 alpha=None, beta=None, gamma=None, phi=None,
                 additive_only=False, lmbda=None,
                 lower=[0.0001, 0.0001, 0.0001, 0.8],
                 upper=[0.9999, 0.9999, 0.9999, 0.98],
                 optimisation_criterion=['lik', 'amse', 'mse', 'sigma', 'mae'],
                 nmse=3, bounds=['both', 'usual', 'admissible'],
                 information_criterion=['aicc', 'aic', 'bic'], restrict=True,
                 allow_multiplicative_trend=False, use_initial_values=False):

        self.model = model
        self.sp = sp
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
        y = self.y_train.to_numpy().tolist()
        orig_y = y
        # Make damped into list
        if isinstance(self.damped, bool):
            self.damped = [self.damped]
        # Set lambda value
        # inherit from fitted model
        if isinstance(self.model, AutoETS) and self.lmbda is None:
            self.lmbda = self.model.lmbda

        # determine lmbda value
        if self.lmbda is None:
            y, self.lmbda = boxcox(y, self.lmbda)
            self.additive_only = True

        # Check for nmse value
        if self.nmse < 1 or self.nmse > 30:
            raise ValueError("nmse out of range (1 <= nmse <=30).")

        # Check for lower and upper limits
        if any(np.greater(self.lower, self.upper)):
            raise ValueError("Lower limits must be less than upper limits.")

        # Refit model to new data if model is an AutoETS object
        if isinstance(self.model, AutoETS):
            # prevent alpha being 0
            self.alpha = max(self.model.alpha, 1e-10)
            self.beta = self.model.beta
            self.gamma = self.model.gamma
            self.phi = self.model.phi

    def _etsmodel(self, y, sp, errortype, trendtype, seasontype, damped,
                  alpha, beta, gamma, phi, lower, upper,
                  optimisation_criterion, nmse, bounds, maxit,
                  control, seed):
        """
        Adapted from ets.R/etsmodel

        Parameters
        ----------
        y: float list
        sp: int
        errortype: str
        trendtype: str
        seasontype: str
        damped: bool
        alpha: float = None
        beta: float = None
        gamma: float = None
        phi: float = None
        lower: float list
        upper: float list
        optimisation_criterion: str
        nmse: int
        bounds: str
        maxit: int = 2000
        control: list = None
        seed: float = None

        Returns
        -------
        [loglik, aic, bic, aicc, mse, amse, fit, residuals, fitted, states,
            par]: list
            [0]loglik: float
            [1]aic: float
            [2]bic: float
            [3]aicc: float
            [4]mse: float
            [5]amse: float
            [6]fit:
            [7]residuals: float list
            [8]fitted: float list
            [9]states: float array
            [10]par: float array

        """
        # Assuming start of time series is 1
        tsp_y = [1, len(y), sp]
        if seasontype != "N":
            m = tsp_y[2]
        else:
            m = 1

        # Modify limits of alpha, beta or gamma that have been specified
        if alpha is not None:
            upper[1] = min(alpha, upper[1])
            upper[2] = min(1 - alpha, upper[2])
        if beta is not None:
            lower[0] = max(beta, lower[0])
        if gamma is not None:
            upper[0] = min(1 - gamma, upper[0])

        # Initialise smoothing parameters
        par = self._initparam(alpha, beta, gamma, phi, trendtype, seasontype,
                              damped, lower, upper, m)
        par_noopt = np.empty(4)
        par_noopt[0] = np.nan if alpha is None else alpha
        par_noopt[1] = np.nan if beta is None else beta
        par_noopt[2] = np.nan if gamma is None else gamma
        par_noopt[3] = np.nan if phi is None else phi
        if not np.isnan(par[0]):
            alpha = par[0]
        if not np.isnan(par[1]):
            beta = par[1]
        if not np.isnan(par[2]):
            gamma = par[2]
        if not np.isnan(par[3]):
            phi = par[3]

        # If errortype == "M" or trendtype == "M" or seasontype == "M"
        # bounds = "usual"
        if self._check_param(alpha, beta, gamma, phi, lower, upper, bounds, m):
            raise ValueError("Parameters out of range.")

        # Initialise state
        init_state = self._initstate(y, trendtype, seasontype)
        nstate = len(init_state)
        par = par.tolist() + init_state
        lower = lower + np.repeat(-np.inf, nstate).tolist()
        upper = upper + np.repeat(np.inf, nstate).tolist()

        n_p = len(par)
        if n_p >= len(y) - 1:  # Not enough data to continue
            # return aic, bic, aicc, mse, amse, fit, par, states
            return [None, np.inf, np.inf, np.inf, np.inf, np.inf, None, None,
                    None, init_state, par]

        """
        To do
        """

        fred = self._etsNelderMead(self, funcPtr, dpar,
                                   np.sqrt(np.finfo(float).eps), maxit)

        fit_par = fred.x
        init_state = fit_par[(n_p - nstate): n_p + 1]
        # Add extra state
        if seasontype != "N":
            init_state = init_state + [m * (seasontype == "M") -
                                       sum(init_state[(1 + (trendtype != "N")):
                                                      nstate + 1])]

        if not np.isnan(fit_par[0]):
            alpha = fit_par[0]
        if not np.isnan(fit_par[1]):
            beta = fit_par[1]
        if not np.isnan(fit_par[2]):
            gamma = fit_par[2]
        if not np.isnan(fit_par[3]):
            phi = fit_par[3]

        e_lik, e_amse, e_e, e_states = self._pegelsresid(y, m, init_state,
                                                         errortype, trendtype,
                                                         seasontype, damped,
                                                         alpha, beta, gamma,
                                                         phi, nmse)

        n_p += 1
        ny = len(y)
        aic = e_lik + 2 * n_p
        bic = e_lik + n_p.log(ny) * n_p
        aicc = aic + 2 * n_p * (n_p + 1) / (n_p - n_p - 1)

        mse = e_amse[0]
        amse = np.mean(e_amse)

        states = e_states
        states_start = tsp_y[0] - 1 / tsp_y[2]

        for i in range(4):
            fit_par[i] = fit_par[i] if np.isnan(par_noopt[i]) else par_noopt[i]
        if errortype == "A":
            fits = y - e_e
        else:
            fits = y / (1 + e_e)
        fitted = fits
        return [-0.5 * e_lik, aic, bic, aicc, mse, amse, fred, residuals,
                fitted, states, fit_par]

    def _etsNelderMead(self, funcPtr, dpar, tol, maxit):
        """
        Adapted from etsTargetFunctionWrapper.cpp/etsNelderMead
            The documentation of nmmin() function can be found at:
            https://cran.r-project.org/doc/manuals/r-release/R-exts.html, and
            https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html

        scipy.optimize.minimize includes the same alpha, beta and gamma vlaues
        (reflection, contraction and expansion factors in Nelder-Mead method).

        Parameters
        ----------
        var: float array
        tol: float
        maxit: int

        Returns
        -------
        res: OptimizeResult object
        """

        res = minimize(fun=funcPtr, x0=dpar, method='Nelder-Mead',
                       options={'maxiter': maxit, 'xatol': tol, 'fatol': tol})

        return res

    def _ETsTargetFunction(self, y, nstate, errortype, trendtype, seasontype,
                           damped, lower, upper, optimisation_criterion,
                           nmse, bounds, m, optAlpha, optBeta, optGamma,
                           optPhi, givenAlpha, givenBeta, givenGamma,
                           givenPhi, alpha, beta, gamma, phi, par):
        """
        Adapted from etsTargetFunction.cpp

        Parameters
        ----------
        y: list
        nstate: int
        errortype: int
        trendtype: int
        seasontype: int
        damped: bool
        lower: list
        uppder: list
        optimisation_criterion: str
        nmse: int
        bounds: str
        m: int
        optAlpha: bool
        optBeta: bool
        optGamma: bool
        optPhi: bool
        givenAlpha: bool
        givenBeta: bool
        givenGamma: bool
        givenPhi: bool
        alpha: float
        beta: float
        gamma: float
        phi: float
        par: float array

        Returns
        -------
        """

        lik = 0
        objval = 0
        amse = np.zeros(30)
        e = np.zeros(len(y))
        state = []
        n = len(y)

        def admissible():
            nonlocal alpha, beta, gamma, phi, m, \
                optBeta, givenBeta, optGamma, givenGamma

            if phi < 0 or phi > 1 + 1e-8:
                return False

            # If gamma was set by the user or is optimised,
            # the bounds need to be enforced
            if not optGamma and not givenGamma:
                if alpha < 1 - 1 / phi or alpha > 1 + 1 / phi:
                    return False
                if optBeta or givenBeta:
                    if beta < alpha * (phi - 1) or beta > (1 + phi) * (2 -
                                                                       alpha):
                        return False
            elif m > 1:  # Seasonal model
                if not optBeta and not givenBeta:
                    beta = 0

                d = max(1 - 1 / phi - alpha, 0)
                if gamma < d or gamma > 1 + 1 / phi - alpha:
                    return False
                if alpha < 1 - 1 / phi - gamma * (1 - m + phi * m) / (2 * phi
                                                                      * m):
                    return False
                if beta < - (1 - phi) * (gamma / m + alpha):
                    return False

                # End of easy tests. Now use characteristic equation
                opr = [1, alpha + beta - phi]
                for i in range(m - 2):
                    opr += [alpha + beta - alpha * phi]
                opr += [alpha + beta - alpha * phi + gamma - 1]
                opr += [phi * (1 - alpha - gamma)]

                # Modify cpolyroot()
                # Reverse order of opr from decreasing to increasing powers
                opr = opr[::-1]
                # Obtain roots
                roots = np.polynomial.polynomial.polyroots(opr)
                # Separate real and imaginary part of the roots
                zeror = roots.real
                zeroi = roots.imag

                max_val = 0
                for i in range(len(zeror)):
                    abs_val = np.sqrt(zeror[i] * zeror[i] +
                                      zeroi[i] * zeroi[i])
                    if abs_val > max_val:
                        max_val = abs_val

                if max_val > 1 + 1e-10:
                    return False

            # Passed all tests
            return True

        def check_params():
            nonlocal bounds, alpha, beta, gamma, phi, lower, upper,\
                optAlpha, optBeta, optGamma, optPhi

            if bounds != "admissible":
                if optAlpha:
                    if alpha < lower[0] or alpha > upper[0]:
                        return False
                if optBeta:
                    if beta < lower[1] or beta > alpha or beta > upper[1]:
                        return False
                if optPhi:
                    if phi < lower[3] or phi > upper[3]:
                        return False
                if optGamma:
                    if gamma < lower[2] or gamma > 1 - alpha \
                            or gamma > upper[2]:
                        return False
            if bounds != "usual":
                if not admissible():
                    return False
            return True

        def eval(p_par):
            nonlocal par, alpha, beta, gamma, phi, nstate, objval, state, m, \
                optAlpha, optBeta, optGamma, optPhi, y, e, lik, amse, n, \
                seasontype, trendtype, errortype, optimisation_criterion, nmse
            # Check if the parameter configuration has changed. If not, return
            if (par == p_par).all():
                return

            par = p_par

            if optAlpha:
                alpha = par[0]
            if optBeta:
                beta = par[1]
            if optGamma:
                gamma = par[2]
            if optPhi:
                phi = par[3]

            if not check_params():
                objval = np.inf
                return

            for i in range(len(par) - nstate, len(par)):
                state += [par[i]]

            # Add extra state
            if seasontype != 0:  # "N" = 0, "M" = 2
                sum = 0
                for i in range(1 + 1 if trendtype != 0 else 0, nstate):
                    sum += state[i]

                new_state = m * 1 if seasontype == 2 else 0 - sum

                state += [new_state]

            # Check states
            if seasontype == 2:
                min = np.inf
                start = 1
                if trendtype != 0:
                    start = 2
                for i in range(start, len(state)):
                    if state[i] < min:
                        min = state[i]
                if min < 0:
                    objval = np.inf

            p = len(state)
            for i in range(p * len(y) + 1):
                state += [0]

            state, e, lik, amse = self._ets_calculation(y, n, state, m,
                                                        errortype,
                                                        trendtype, seasontype,
                                                        alpha, beta, gamma,
                                                        phi, e, lik, amse,
                                                        nmse)

            # Avoid perfect fits
            if lik < -1e10:
                lik = -1e10

            if np.isnan(lik):
                lik = np.inf

            if abs(lik + 99999) < 1e-7:
                lik = np.inf

            if optimisation_criterion == "lik":
                objval = lik
            elif optimisation_criterion == "mse":
                objval = amse[0]
            elif optimisation_criterion == "amse":
                mean = 0
                for i in range(nmse):
                    mean += amse[i] / nmse
                objval = mean
            elif optimisation_criterion == "sigma":
                mean = 0
                n_e = len(e)
                for i in range(n_e):
                    mean += e[i] * e[i] / n_e
                objval = mean
            elif optimisation_criterion == "mae":
                mean = 0
                n_e = len(e)
                for i in range(n_e):
                    mean += abs(e[i]) / n_e
                objval = mean

        eval(par)
        return objval

    def _etsTargetFunction(self, par, y, nstate, errortype, trendtype,
                           seasontype,  damped, par_noopt, lowerb, upperb,
                           optimisation_criterion, nmse, bounds, m):
        """
        Adapted from ets.R/etsTargetFunctionInit

        Parameters
        ----------
        par: float array shape(1, 4)
        y: float list
        nstate: int
        errortype: str
        trendtype: str
        seasontype: str
        damped: bool
        par_noopt: float array shape(1, 4)
        lowerb: float list
        upperb: float list
        optimisation_criterion: str
        nmse: int
        bounds: str
        m: int

        Returns
        -------
        """
        alpha = par[0] if np.isnan(par_noopt[0]) else par_noopt[0]
        if np.isnan(alpha):
            raise ValueError("Alpha value error.")
        if trendtype != "N":
            beta = par[1] if np.isnan(par_noopt[1]) else par_noopt[1]
            if np.isnan(beta):
                raise ValueError("Beta value error.")
        else:
            beta = None
        if seasontype != "N":
            gamma = par[2] if np.isnan(par_noopt[2]) else par_noopt[2]
            if np.isnan(gamma):
                raise ValueError("Gamma value error.")
        else:
            m = 1
            gamma = None
        if damped:
            phi = par[3] if np.isnan(par_noopt[3]) else par_noopt[3]
            if np.isnan(phi):
                raise ValueError("Phi value error.")
        else:
            phi = None

        # Determine which values to optimise and which ones are given by user
        optAlpha = alpha is not None
        optBeta = beta is not None
        optGamma = gamma is not None
        optPhi = phi is not None

        givenAlpha = givenBeta = givenGamma = givenPhi = False

        if not np.isnan(par_noopt[0]):
            optAlpha = False
            givenAlpha = True
        if not np.isnan(par_noopt[1]):
            optBeta = False
            givenBeta = True
        if not np.isnan(par_noopt[2]):
            optGamma = False
            givenGamma = True
        if not np.isnan(par_noopt[3]):
            optPhi = False
            givenPhi = True

        if not damped:
            phi = 1
        if trendtype == "N":
            beta = 0
        if seasontype == "N":
            gamma = 0

        if errortype == "A":
            error = 1
        else:  # errortype == "M"
            error = 2

        if trendtype == "N":
            trend = 0
        elif trendtype == "A":
            trend = 1
        else:  # trendtype == "M"
            trend = 2

        if seasontype == "N":
            season = 0
        elif seasontype == "A":
            season = 1
        else:  # seasontype =="M"
            season = 2

        res = self._ETsTargetFunction(y, nstate, error, trend, season, damped,
                                      lowerb, upperb, optimisation_criterion,
                                      nmse, bounds, m, optAlpha, optBeta,
                                      optGamma, optPhi, givenAlpha, givenBeta,
                                      givenGamma, givenPhi, alpha, beta, gamma,
                                      phi, par)

        return res

    def _initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped,
                   lower, upper, m):
        """
        Adapted from ets.R/initparam

        Parameters
        ----------
        alpha: float
        beta: float
        gamma: float
        phi: float
        trendtype: str
        seasontype: str
        damped: bool
        lower: float list
        upper: float list
        m: int

        Returns
        -------
        par: float array shape(1, 4)
            If any parameter is not calculated, NaN is returned.
        """
        if any(np.greater(lower, upper)):
            raise ValueError("Lower limits must be less than upper limits.")

        par = np.empty(4)
        par[:] = np.nan

        # Select alpha
        if alpha is None:
            alpha = lower[0] + 0.2 * (upper[0] - lower[0]) / m
            if alpha > 1 or alpha < 0:
                alpha = lower[0] + 2e-3
            par[0] = alpha

        # Select beta
        if trendtype != "N" and beta is None:
            # Ensure beta > alpha
            upper[1] = min(upper[1], alpha)
            beta = lower[1] + 0.1 * (upper[1] - lower[1])
            if beta < 0 or beta > alpha:
                beta = alpha - 1e-3
            par[1] = beta

        # Select gamma
        if seasontype != "N" and gamma is None:
            # Ensure gamma < 1 - alpha
            upper[2] = min(upper[2], 1 - alpha)
            gamma = lower[2] + 0.05 * (upper[2] + lower[2])
            if gamma < 0 or gamma > 1 - alpha:
                gamma = 1 - alpha - 1e-3
            par[2] = gamma

        # Select phi
        if damped and phi is None:
            phi = lower[3] + 0.99 * (upper[3] - lower[3])
            if phi < 0 or phi > 1:
                phi = upper[3] - 1e-3
            par[3] = phi

        return par

    def _check_param(alpha, beta, gamma, phi, lower, upper, bounds, m):
        """
        Adapted from ets.R/check.param

        Parameters
        ----------
        alpha: float
        beta: float
        gamma: float
        phi: float
        lower: float list
        upper: float list
        bounds: str
        m: int

        Returns
        -------
        Bool (0 or 1)
        """
        def _admissible(alpha, beta, gamma, phi, m):
            """
            Adapted from ets.R/admissible

            Parameters
            ----------
            alpha: float
            beta: float
            gamma: float
            phi: float
            m: int

            Returns
            -------
            Bool (0 or 1)
            """
            if phi is None:
                phi = 1
            if phi < 0 or phi > 1 + 1e-8:
                return 0

            if gamma is None:
                if alpha < 1 - 1 / phi or alpha > 1 + 1 / phi:
                    return 0
                if beta is not None:
                    if beta < alpha * (phi - 1) or beta > (1 + phi) * (2 -
                                                                       alpha):
                        return 0

            elif m > 1:  # Seasonal model
                if beta is None:
                    beta = 0
                if gamma < max(1 - 1 / phi - alpha, 0) or gamma > 1 + 1 / phi \
                        - alpha:
                    return 0
                if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / \
                        (2 * phi * m):
                    return 0
                if beta < -(1 - phi) * (gamma / m + alpha):
                    return 0
            # End of easy tests. Now using characteristic equation
            P = [phi * (1 - alpha - gamma), alpha + beta - alpha * phi + gamma
                 - 1] + np.repeat(alpha + beta - alpha * phi, m - 2).tolist() \
                + [alpha + beta - phi, 1]
            roots = np.polynomial.polynomial.polyroots(P)

            if max(abs(roots)) > 1 + 1e-10:
                return 0

            # Passed all tests
            return 1
        # ========================
        if bounds != "admissible":
            if alpha is not None:
                if alpha < lower[0] or alpha > upper[0]:
                    return 0
            if beta is not None:
                if beta < lower[1] or beta > upper[1]:
                    return 0
            if gamma is not None:
                if gamma < lower[2] or gamma > upper[2]:
                    return 0
            if phi is not None:
                if phi < lower[3] or phi > upper[3]:
                    return 0
        if bounds != "usual":
            if not _admissible(alpha, beta, gamma, phi, m):
                return 0
        return 1

    def _initstate(y, freqeuncy, trendtype, seasontype):
        """
        Adapted from ets.R/initstate

        Parameters
        ----------
        y: float list
        sp: int
        trendtype: str
        seastype: str

        Returns
        -------
        [l0, b0, init_seas]: list
            l0: int
            b0: int
            init_seas: list
        """

        def _fourier(y, sp):
            """
            Adapted from season.R/...fourier (forecast package)
            Performs similarly to fourier function in R,
            with K = 1 and h = NULL

            Parameters
            ----------
            y: float list
            sp: int

            Returns
            -------
            X: float array shape(len(y), 2)
                first column is sine terms
                second column is cosine terms
            """
            # Compute matrix of Fourier terms
            n = len(y)
            times = np.linspace(1, n, n)
            sp = 12
            p = 1 / sp
            X = np.zeros((n, 2))
            X[:, 0] = np.sin(2 * p * times * np.pi)
            X[:, 1] = np.cos(2 * p * times * np.pi)
            return X

        if seasontype != "N":
            # Do decomposition
            m = freqeuncy
            n = len(y)
            if n < 4:
                raise ValueError("Not enough time series data.")
            elif n < 3 * m:
                # Fit simple Fourier model
                fouriery = _fourier(y, 12)
                trendy = np.linspace(1, n, n)
                mod = smf.ols(formula='y ~ trendy + fouriery', data=y)
                res = mod.fit()
                if seasontype == "A":
                    y_d = y - res.params[0] - res.params[1] * range(1, n + 1)
                # seasontype == "M". Biased method, only need starting point
                else:
                    y_d = y / (res.params[0] + res.params[1] * range(1, n + 1))
            else:  # n is large enough to do a decomposition
                if seasontype == "A":
                    model = "additive"
                else:
                    model = "multiplicative"
                res = decompose(y, model=model, period=n / m)
                y_d = res.seasonal

            # initial seasonal component
            init_seas = y_d[1: m][::-1].to_numpy()

            # Seasonally adjusted data
            if seasontype == "A":
                y_sa = y - y_d
            else:
                # We do not want negative seasonal indexes
                init_seas = np.maximum(init_seas, 1e-2)
                if sum(init_seas) > m:
                    init_seas = init_seas / sum(init_seas + 1e-2)
                y_sa = y / np.maximum(y_d, 1e-2)
        else:  # Non-seasonal model
            m = 1
            init_seas = None
            y_sa = y

        maxn = min(max(10, 2 * m), len(y_sa))

        if trendtype == "N":
            l0 = np.mean(y_sa[0: maxn])
            b0 = None
        else:  # Simple linear regression on seasonally adjusted data
            lsfit = linregress(range(1, maxn + 1), y_sa[0: maxn])
            if trendtype == "A":
                l0 = lsfit.intercept
                b0 = lsfit.slope
                # If error type is "M", then we do not want l0 + b0 = 0
                # So perturb just in case
                if abs(l0 + b0) < 1e-8:
                    l0 = l0 * (1 + 1e-3)
                    b0 = b0 * (1 + 1e-3)
            else:  # if trendtype == "M"
                l0 = lsfit.intercept + lsfit.slope  # First fitted value
                if abs(l0) < 1e-8:
                    l0 = 1e-7
                # Ratio of first two fitted values
                b0 = (lsfit.intercept + 2 * lsfit.slope) / l0
                l0 = l0 / b0  # First fitted value divided by b0
                if abs(b0) > 1e10:  # Avoid infinite slopes
                    b0 = np.sign(b0) * 1e10
                # Simple linear approximation did not work
                if l0 < 1e-8 or b0 < 1e-8:
                    l0 = max(y_sa[0], 1e-3)
                    b0 = max(y_sa[1] / y_sa[0], 1e-3)

        return [l0, b0, init_seas.tolist()]

    # pegelsresid.C function
    def _pegelsresid(self, y, m, init_state, errortype, trendtype, seasontype,
                     damped, alpha, beta, gamma, phi, nmse):
        """
        Adapted from ets.R/pegelsresid.C

        Parameters
        ----------
        y: float list
        m: int
        init_state: list
        errortype: str
        trendtype: str
        seasontype: str
        damped: bool
        alpha: float
        beta: float
        gamma: float
        phi: float
        nmse: int

        Returns
        -------
        lik: float
        amse: float array
        e: float array
        states: float array
        """

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

        if errortype == "A":
            error = 1
        else:  # errortype == "M"
            error = 2

        if trendtype == "N":
            trend = 0
        elif trendtype == "A":
            trend = 1
        else:  # trendtype == "M"
            trend = 2

        if seasontype == "N":
            season = 0
        elif seasontype == "A":
            season = 1
        else:  # seasontype =="M"
            season = 2

        x, e, lik, amse = self._ets_calculation(y, n, x, m, error, trend,
                                                season, alpha, beta, gamma,
                                                phi, e, lik, amse, nmse)

        if not np.isnan(lik):
            if abs(lik + 99999) < 1e-7:
                lik = np.nan

        return lik, amse, e, np.array(x.reshape(n + 1, p))

    """
    The following functions:

    _ets_calculation
    _forecast
    _update

    are adapted from the etscalc.c file, available at:
    https://github.com/robjhyndman/forecast/blob/master/src/etscalc.c
    """
    def _ets_calculation(self, y, n, x, m, error, trend, season, alpha, beta,
                         gamma, phi, e, lik, amse, nmse):
        """
        Adapted from etscalc.c/etscalc

        Parameters
        ----------
        y: float
        n: int
        x: float
        m: int
        error: int
        trend: int
        season: int
        alpha: float
        beta: float
        gamma: float
        phi: float
        e: float
        lik: float
        amse: float
        nmse: int

        Returns
        -------
        x: float array
        e: float array
        lik: float
        amse: float array
        """

        def _forecast(L, b, s, m, trend, season, phi, f, h):
            """
            Adapted from etscalc.c/forecast

            Parameters
            ----------
            L(l): float
            b: float
            s: float
            m: int
            trend: int
            season: int
            phi: float
            f: float array
            h: int

            Returns
            -------
            f: float array
            """
            phi_star = phi

            # Forecasts
            for i in range(h):
                if trend == 0:
                    f[i] = L
                elif trend == 1:
                    f[i] = L + phi_star * b
                elif b < 0:
                    f[i] = - 99999.0
                else:
                    f[i] = L * pow(b, phi_star)
                j = m - 1 - i
                while j < 0:
                    j += m
                if season == 1:
                    f[i] = f[i] + s[j]
                elif season == 2:
                    f[i] = f[i] * s[j]
                if i < (h - 1):
                    if abs(phi - 1) < 1.0e-10:
                        phi_star = phi_star + 1.0
                    else:
                        phi_star = phi_star + pow(phi, (i+1))
            return f

        def _update(oldl, L, oldb, b, olds, s, m, trend, season,
                    alpha, beta, gamma, phi, y):
            """
            Adapted from etscalc.c/update

            Parameters
            ----------
            oldl: float
            L(l): float
            oldb: float
            b: float
            olds: float
            s: float
            m: int
            trend: int
            season: int
            alpha: float
            beta: float
            gamma: float
            phi: float
            y: float

            Returns
            -------
            L: float
            b: float
            s: float array
            """
            # New level
            if trend == 0:
                q = oldl  # l(t - 1)
                phib = 0
            elif trend == 1:
                phib = phi * oldb
                q = oldl + phib  # l(t - 1) + phi * b(t - 1)
            elif abs(phi - 1.0) < 1.0e-10:
                phib = oldb
                q = oldl * oldb  # l(t -1) * b(t - 1)
            else:
                phib = pow(oldb, phi)
                q = oldl * phib  # l(t - 1) * b(t - 1) ^ phi

            if season == 0:
                p = y
            elif season == 1:
                p = y - olds[m - 1]  # y[t] - s[t - m]
            else:
                if abs(olds[m - 1]) < 1.0e-10:
                    p = 1.0e10
                else:
                    p = y / olds[m - 1]  # y[t] / s[t - m]
            L = q + alpha * (p - q)

            # New growth
            if trend > 0:
                if trend == 1:
                    r = L - oldl  # l[t] - l[t - 1]
                else:  # if trend == 2
                    if abs(oldl) < 1.0e-10:
                        r = 1.0e10
                    else:
                        r = L / oldl  # l[t] / l[t - 1]
                # b[t] = phi * b[t - 1] + beta * (r - phi * b[t - 1])
                # b[t] = b[t - 1] ^ phi + beta * (r - b[t - 1] ^ phi)
                b = phib + (beta / alpha) * (r - phib)

            # New season
            if trend > 0:
                if season == 1:
                    t = y - q
                else:  # if season == 2
                    if abs(q) < 1.0e-10:
                        t = 1.0e10
                    else:
                        t = y / q
                # s[t] = s[t - m] + gamma * (t - s[t - m])
                s[0] = olds[m - 1] + gamma * (t - olds[m - 1])
                for j in range(m):
                    s[j] = olds[j - 1]  # s[t] = s[t]

            return L, b, s
        # =================
        olds = np.zeros(24)
        s = np.zeros(24)
        f = np.zeros(24)
        denom = np.zeros(30)
        oldl = L = oldb = b = lik2 = tmp = nstates = 0

        # Check for m value
        if m > 24 and season > 0:
            return
        elif m < 1:
            m = 1

        # Check for nmse value
        if nmse > 30:
            nmse = 30

        nstates = m * (season > 0) + 1 + (trend > 0)

        # Copy initial state components
        L = x[0]
        if trend > 0:
            b = x[1]
        if season > 0:
            for j in range(m):
                s[j] = x[(trend > 0) + j + 1]

        lik = 0.0
        lik2 = 0.0
        for j in range(nmse):
            amse[j] = 0.0
            denom[j] = 0.0

        for i in range(n):
            # Copy previous state
            oldl = L
            if trend > 0:
                oldb = b
            if season > 0:
                for j in range(m):
                    olds[j] = s[j]

            # One step forecast
            f = _forecast(oldl, oldb, olds, m, trend, season, phi, f, nmse)
            if abs(f[0] - (-99999)) < 1.0e-10:
                lik = -99999.0
                return

            if error == 1:
                e[i] = y[i] - f[0]
            else:
                e[i] = (y[i] - f[0]) / f[0]
            for j in range(nmse):
                if i + j < n:
                    denom[j] += 1.0
                    tmp = y[i+j] - f[j]
                    amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) \
                        / denom[j]

            # Update state
            L, b, s = _update(oldl, L, oldb, b, olds, s, m, trend, season,
                              alpha, beta, gamma, phi, y[i])

            # Store new state
            x[nstates * (i + 1)] = L
            if trend > 0:
                x[nstates * (i + 1) + 1] = b
            if season > 0:
                for j in range(m):
                    x[(trend > 0) + nstates * (i + 1) + j + 1] = s[j]
            lik = lik + e[i] * e[i]
            lik2 += np.log(abs(f[0]))

        lik = n * np.log(lik)
        if error == 2:
            lik += 2 * lik2

        return x, e, lik, amse
