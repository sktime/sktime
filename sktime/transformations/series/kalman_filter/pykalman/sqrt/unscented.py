'''
=========================================
Inference for Non-Linear Gaussian Systems
=========================================

This module contains "Square Root" implementations to the Unscented Kalman
Filter.  Square Root implementations typically propagate the mean and Cholesky
factorization of the covariance matrix in order to prevent numerical error.
When possible, Square Root implementations should be preferred to their
standard counterparts.

References
----------

* Terejanu, G.A. Towards a Decision-Centric Framework for Uncertainty
  Propagation and Data Assimilation. 2010. Page 108.
* Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman Filter for
  State and Parameter-Estimation. 2001.
'''
import numpy as np
from numpy import ma
from scipy import linalg

from ..utils import array1d, array2d, check_random_state

from ..standard import _last_dims, _arg_or_default
from ..unscented import AdditiveUnscentedKalmanFilter as AUKF, \
    SigmaPoints, Moments


def _reconstruct_covariances(covariance2s):
    '''Reconstruct covariance matrices given their cholesky factors'''
    if len(covariance2s.shape) == 2:
        covariance2s = covariance2s[np.newaxis, :, :]

    T = covariance2s.shape[0]
    covariances = np.zeros(covariance2s.shape)

    for t in range(T):
        M = covariance2s[t]
        covariances[t] = M.T.dot(M)

    return covariances


def cholupdate(A2, X, weight):
    '''Calculate chol(A + w x x')

    Parameters
    ----------
    A2 : [n_dim, n_dim] array
        A = A2.T.dot(A2) for A positive definite, symmetric
    X : [n_dim] or [n_vec, n_dim] array
        vector(s) to be used for x.  If X has 2 dimensions, then each row will be
        added in turn.
    weight : float
        weight to be multiplied to each x x'. If negative, will use
        sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

    Returns
    -------
    A2 : [n_dim, n_dim array]
        cholesky decomposition of updated matrix

    Notes
    -----

    Code based on the following MATLAB snippet taken from Wikipedia on
    August 14, 2012::

        function [L] = cholupdate(L,x)
            p = length(x);
            x = x';
            for k=1:p
                r = sqrt(L(k,k)^2 + x(k)^2);
                c = r / L(k, k);
                s = x(k) / L(k, k);
                L(k, k) = r;
                L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
                x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
            end
        end
    '''
    # make copies
    X = X.copy()
    A2 = A2.copy()

    # standardize input shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n_vec, n_dim = X.shape

    # take sign of weight into account
    sign, weight = np.sign(weight), np.sqrt(np.abs(weight))
    X = weight * X

    for i in range(n_vec):
        x = X[i, :]
        for k in range(n_dim):
            r_squared = A2[k, k] ** 2 + sign * x[k] ** 2
            r = 0.0 if r_squared < 0 else np.sqrt(r_squared)
            c = r / A2[k, k]
            s = x[k] / A2[k, k]
            A2[k, k] = r
            A2[k, k + 1:] = (A2[k, k + 1:] + sign * s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * A2[k, k + 1:]
    return A2


def qr(A):
    '''Get square upper triangular matrix of QR decomposition of matrix A'''
    N, L = A.shape
    if not N >= L:
        raise ValueError("Number of columns must exceed number of rows")
    Q, R = linalg.qr(A)
    return R[:L, :L]


def points2moments(points, sigma2_noise=None):
    '''Calculate estimated mean and covariance of sigma points

    Parameters
    ----------
    points : [2 * n_dim_state + 1, n_dim_state] SigmaPoints
        SigmaPoints object containing points and weights
    sigma_noise : [n_dim_state, n_dim_state] array
        additive noise covariance matrix, if any

    Returns
    -------
    moments : Moments object of size [n_dim_state]
        Mean and covariance estimated using points
    '''
    (points, weights_mu, weights_sigma) = points
    mu = points.T.dot(weights_mu)

    # make points to perform QR factorization on. each column is one data point
    qr_points = [
        np.sign(weights_sigma)[np.newaxis, :]
        * np.sqrt(np.abs(weights_sigma))[np.newaxis, :]
        * (points.T - mu[:, np.newaxis])
    ]
    if sigma2_noise is not None:
        qr_points.append(sigma2_noise)
    sigma2 = qr(np.hstack(qr_points).T)
    #sigma2 = cholupdate(sigma2, points[0] - mu, weights_sigma[0])
    return Moments(mu.ravel(), sigma2)


def moments2points(moments, alpha=None, beta=None, kappa=None):
    '''Calculate "sigma points" used in Unscented Kalman Filter

    Parameters
    ----------
    moments : [n_dim] Moments object
        mean and covariance of a multivariate normal
    alpha : float
        Spread of the sigma points. Typically 1e-3.
    beta : float
        Used to "incorporate prior knowledge of the distribution of the state".
        2 is optimal is the state is normally distributed.
    kappa : float
        a parameter which means ????

    Returns
    -------
    points : [2*n_dim+1, n_dim] SigmaPoints
        sigma points and associated weights
    '''
    (mu, sigma2) = moments
    n_dim = len(mu)
    mu = array2d(mu, dtype=float)

    if alpha is None:
      alpha = 1.0
    if beta is None:
      beta = 0.0
    if kappa is None:
      kappa = 3.0 - n_dim

    # just because I saw it in the MATLAB implementation
    sigma2 = sigma2.T

    # Calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    c = n_dim + lamda

    # calculate the sigma points; that is,
    #   mu
    #   mu + each column of sigma2 * sqrt(c)
    #   mu - each column of sigma2 * sqrt(c)
    # Each column of points is one of these.
    points = np.tile(mu.T, (1, 2 * n_dim + 1))
    points[:, 1:(n_dim + 1)] += sigma2 * np.sqrt(c)
    points[:, (n_dim + 1):] -= sigma2 * np.sqrt(c)

    # Calculate weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    return SigmaPoints(points.T, weights_mean, weights_cov)


def _unscented_transform(points, f=None, points_noise=None, sigma2_noise=None):
    '''Apply the Unscented Transform.

    Parameters
    ==========
    points : [n_points, n_dim_1] array
        points representing state to pass through `f`
    f : [n_dim_1, n_dim_3] -> [n_dim_2] function
        function to apply pass all points through
    points_noise : [n_points, n_dim_3] array
        points representing noise to pass through `f`, if any.
    sigma2_noise : [n_dim_2, n_dim_2] array
        square root of covariance matrix for additive noise

    Returns
    =======
    points_pred : [n_points, n_dim_2] array
        points passed through f
    mu_pred : [n_dim_2] array
        empirical mean
    sigma2_pred : [n_dim_2, n_dim_2] array
        R s.t. R' R = empirical covariance
    '''
    n_points, n_dim_state = points.points.shape
    (points, weights_mean, weights_covariance) = points

    # propagate points through f.  Each column is a sample point
    if f is not None:
        if points_noise is None:
            points_pred = [f(points[i]) for i in range(n_points)]
        else:
            points_pred = [f(points[i], points_noise[i]) for i in range(n_points)]
    else:
        points_pred = points

    # make each row a predicted point
    points_pred = np.vstack(points_pred)
    points_pred = SigmaPoints(points_pred, weights_mean, weights_covariance)

    # calculate approximate mean, covariance
    moments_pred = points2moments(
        points_pred, sigma2_noise=sigma2_noise
    )

    return (points_pred, moments_pred)


def _unscented_correct(cross_sigma, moments_pred, obs_moments_pred, z):
    '''Correct predicted state estimates with an observation

    Parameters
    ----------
    cross_sigma : [n_dim_state, n_dim_obs] array
        cross-covariance between the state at time t given all observations
        from timesteps [0, t-1] and the observation at time t
    moments_pred : [n_dim_state] Moments
        mean and covariance of state at time t given observations from
        timesteps [0, t-1]
    obs_moments_pred : [n_dim_obs] Moments
        mean and covariance of observation at time t given observations from
        times [0, t-1]
    z : [n_dim_obs] array
        observation at time t

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean and covariance of state at time t given observations from time
        steps [0, t]
    '''
    mu_pred, sigma2_pred = moments_pred
    obs_mu_pred, obs_sigma2_pred = obs_moments_pred

    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(z)):
        ##############################################
        # Same as this, but more stable (supposedly) #
        ##############################################
        # K = cross_sigma.dot(
        #     linalg.pinv(
        #         obs_sigma2_pred.T.dot(obs_sigma2_pred)
        #     )
        # )
        ##############################################

        # equivalent to this MATLAB code
        # K = (cross_sigma / obs_sigma2_pred.T) / obs_sigma2_pred
        K = linalg.lstsq(obs_sigma2_pred, cross_sigma.T)[0]
        K = linalg.lstsq(obs_sigma2_pred.T, K)[0]
        K = K.T

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        U = K.dot(obs_sigma2_pred)
        sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma2_filt = sigma2_pred
    return Moments(mu_filt, sigma2_filt)


def unscented_filter_predict(transition_function, points_state,
                             points_transition=None,
                             sigma2_transition=None):
    """Predict next state distribution

    Using the sigma points representing the state at time t given observations
    from time steps 0...t, calculate the predicted mean, covariance, and sigma
    points for the state at time t+1.

    Parameters
    ----------
    transition_function : function
        function describing how the state changes between times t and t+1
    points_state : [2*n_dim_state+1, n_dim_state] SigmaPoints
        sigma points corresponding to the state at time step t given
        observations from time steps 0...t
    points_transition : [2*n_dim_state+1, n_dim_state] SigmaPoints
        sigma points corresponding to the noise in transitioning from time step
        t to t+1, if available. If not, assumes that noise is additive
    sigma_transition : [n_dim_state, n_dim_state] array
        covariance corresponding to additive noise in transitioning from time
        step t to t+1, if available. If not, assumes noise is not additive.

    Returns
    -------
    points_pred : [2*n_dim_state+1, n_dim_state] SigmaPoints
        sigma points corresponding to state at time step t+1 given observations
        from time steps 0...t. These points have not been "standardized" by the
        unscented transform yet.
    moments_pred : [n_dim_state] Moments
        mean and covariance corresponding to time step t+1 given observations
        from time steps 0...t
    """
    assert points_transition is not None or sigma2_transition is not None, \
        "Your system is noiseless? really?"
    (points_pred, moments_pred) = (
        _unscented_transform(
            points_state, transition_function,
            points_noise=points_transition, sigma2_noise=sigma2_transition
        )
    )
    return (points_pred, moments_pred)


def unscented_filter_correct(observation_function, moments_pred,
                             points_pred, observation,
                             points_observation=None,
                             sigma2_observation=None):
    """Integrate new observation to correct state estimates

    Parameters
    ----------
    observation_function : function
        function characterizing how the observation at time t+1 is generated
    moments_pred : [n_dim_state] Moments
        mean and covariance of state at time t+1 given observations from time
        steps 0...t
    points_pred : [2*n_dim_state+1, n_dim_state] SigmaPoints
        sigma points corresponding to moments_pred
    observation : [n_dim_state] array
        observation at time t+1. If masked, treated as missing.
    points_observation : [2*n_dim_state, n_dim_obs] SigmaPoints
        sigma points corresponding to predicted observation at time t+1 given
        observations from times 0...t, if available. If not, noise is assumed
        to be additive.
    sigma_observation : [n_dim_obs, n_dim_obs] array
        covariance matrix corresponding to additive noise in observation at
        time t+1, if available. If missing, noise is assumed to be non-linear.

    Returns
    -------
    moments_filt : [n_dim_state] Moments
        mean and covariance of state at time t+1 given observations from time
        steps 0...t+1
    """
    # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
    (obs_points_pred, obs_moments_pred) = (
        _unscented_transform(
            points_pred, observation_function,
            points_noise=points_observation, sigma2_noise=sigma2_observation
        )
    )

    # Calculate Cov(x_t, z_t | z_{0:t-1})
    sigma_pair = (
        ((points_pred.points - moments_pred.mean).T)
        .dot(np.diag(points_pred.weights_mean))
        .dot(obs_points_pred.points - obs_moments_pred.mean)
    )

    # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
    moments_filt = _unscented_correct(sigma_pair, moments_pred, obs_moments_pred, observation)
    return moments_filt


def _additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    '''Apply the Unscented Kalman Filter with additive noise

    Parameters
    ----------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and outputs
        the next.
    g : function or [T] array of functions
        observation function(s). Takes in the current state and outputs the
        current observation.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix
    R : [n_dim_state, n_dim_state] array
        observation covariance matrix

    Returns
    -------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times [0,
        t]
    sigma2_filt : [T, n_dim_state, n_dim_state] array
        sigma2_filt[t] = square root of the covariance of state at time t given
        observations from times [0, t]
    '''
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma2_filt = np.zeros((T, n_dim_state, n_dim_state))
    Q2 = linalg.cholesky(Q)
    R2 = linalg.cholesky(R)

    for t in range(T):
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu, sigma2 = mu_0, linalg.cholesky(sigma_0)
        else:
            mu, sigma2 = mu_filt[t - 1], sigma2_filt[t - 1]

        points_state = moments2points(Moments(mu, sigma2))

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, moments_pred) = (
                unscented_filter_predict(
                    transition_function, points_state, sigma2_transition=Q2
                )
            )
            points_pred = moments2points(moments_pred)

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma2_filt[t] = unscented_filter_correct(
            observation_function, moments_pred, points_pred,
            Z[t], sigma2_observation=R2
        )

    return (mu_filt, sigma2_filt)


def _additive_unscented_smoother(mu_filt, sigma2_filt, f, Q):
    '''Apply the Unscented Kalman Filter assuming additiven noise

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_2filt : [T, n_dim_state, n_dim_state] array
        sigma2_filt[t] = square root of the covariance of state at time t given
        observations from times [0, t]
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and outputs
        the next.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of state at time t given observations from times
        [0, T-1]
    sigma2_smooth : [T, n_dim_state, n_dim_state] array
        sigma2_smooth[t] = square root of the covariance of state at time t
        given observations from times [0, T-1]
    '''
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma2_smooth = np.zeros(sigma2_filt.shape)
    mu_smooth[-1], sigma2_smooth[-1] = mu_filt[-1], sigma2_filt[-1]
    Q2 = linalg.cholesky(Q)

    for t in reversed(range(T - 1)):
        # get sigma points for state
        mu = mu_filt[t]
        sigma2 = sigma2_filt[t]

        moments_state = Moments(mu, sigma2)
        points_state = moments2points(moments_state)

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        transition_function = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = (
            _unscented_transform(points_state, transition_function, sigma2_noise=Q2)
        )

        # Calculate Cov(x_{t+1}, x_t | z_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean).T
            .dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean).T
        )

        # compute smoothed mean, covariance

        #############################################
        # Same as this, but more stable (supposedly)#
        #############################################
        # smoother_gain = (
        #     sigma_pair.dot(linalg.pinv(sigma2_pred.T.dot(sigma2_pred)))
        # )
        #############################################
        smoother_gain = linalg.lstsq(moments_pred.covariance.T, sigma_pair.T)[0]
        smoother_gain = linalg.lstsq(moments_pred.covariance, smoother_gain)[0]
        smoother_gain = smoother_gain.T

        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - moments_pred.mean)
        )
        U = cholupdate(moments_pred.covariance, sigma2_smooth[t + 1], -1.0)
        sigma2_smooth[t] = (
            cholupdate(sigma2_filt[t], smoother_gain.dot(U.T).T, -1.0)
        )

    return (mu_smooth, sigma2_smooth)


class AdditiveUnscentedKalmanFilter(AUKF):
    r'''Implements the Unscented Kalman Filter with additive noise.
    Observations are assumed to be generated from the following process,

    .. math::

        x_0       &\sim \text{Normal}(\mu_0, \Sigma_0)  \\
        x_{t+1}   &=    f_t(x_t) + \text{Normal}(0, Q)  \\
        z_{t}     &=    g_t(x_t) + \text{Normal}(0, R)


    While less general the general-noise Unscented Kalman Filter, the Additive
    version is more computationally efficient with complexity :math:`O(Tn^3)`
    where :math:`T` is the number of time steps and :math:`n` is the size of
    the state space.

    Parameters
    ----------
    transition_functions : function or [n_timesteps-1] array of functions
        transition_functions[t] is a function of the state at time t and
        produces the state at time t+1. Also known as :math:`f_t`.
    observation_functions : function or [n_timesteps] array of functions
        observation_functions[t] is a function of the state at time t and
        produces the observation at time t. Also known as :math:`g_t`.
    transition_covariance : [n_dim_state, n_dim_state] array
        transition noise covariance matrix. Also known as :math:`Q`.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix. Also known as :math:`R`.
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution. Also known as :math:`\mu_0`.
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution. Also known as
        :math:`\Sigma_0`.
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_covariance`, or
        `initial_state_mean`, `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_covariance`.
    random_state : optional, int or RandomState
        seed for random sample generation
    '''
    def filter(self, Z):
        '''Run Unscented Kalman Filter

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t
            given observations from times [0, t]
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, t]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Z.shape[0]

        # run square root filter
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )

        # reconstruct covariance matrices
        filtered_state_covariances = np.zeros(sigma2_filt.shape)
        for t in range(n_timesteps):
            filtered_state_covariances[t] = sigma2_filt[t].T.dot(sigma2_filt[t])

        return (filtered_state_means, filtered_state_covariances)

    def filter_update(self,
                      filtered_state_mean, filtered_state_covariance,
                      observation=None,
                      transition_function=None, transition_covariance=None,
                      observation_function=None, observation_covariance=None):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Parameters
        ----------
        filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t given observations from times
            [1...t]
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t given observations from
            times [1...t]
        observation : [n_dim_obs] array or None
            observation from time t+1.  If `observation` is a masked array and
            any of `observation`'s components are masked or if `observation` is
            None, then `observation` will be treated as a missing observation.
        transition_function : optional, function
            state transition function from time t to t+1.  If unspecified,
            `self.transition_functions` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_function : optional, function
            observation function at time t+1.  If unspecified,
            `self.observation_functions` will be used.
        observation_covariance : optional, [n_dim_obs, n_dim_obs] array
            observation covariance at time t+1.  If unspecified,
            `self.observation_covariance` will be used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t+1 given observations from times
            [1...t+1]
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t+1 given observations
            from times [1...t+1]
        """
        # initialize parameters
        (transition_functions, observation_functions,
         transition_cov, observation_cov,
         _, _) = (
            self._initialize_parameters()
        )

        def default_function(f, arr):
            if f is None:
                assert len(arr) == 1
                f = arr[0]
            return f

        transition_function = default_function(
            transition_function, transition_functions
        )
        observation_function = default_function(
            observation_function, observation_functions
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        # preprocess covariance matrices
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance)
        transition_covariance2 = linalg.cholesky(transition_covariance)
        observation_covariance2 = linalg.cholesky(observation_covariance)

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance2)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = (
            unscented_filter_predict(
                transition_function, points_state,
                sigma2_transition=transition_covariance2
            )
        )
        points_pred = moments2points(moments_pred)

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance2) = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                observation, sigma2_observation=observation_covariance2
            )
        )

        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_covariance2)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, Z):
        '''Run Unscented Kalman Smoother

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t
            given observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            smoothed_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Z.shape[0]

        # run filter, then smoother
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )
        (smoothed_state_means, sigma2_smooth) = (
            _additive_unscented_smoother(
                filtered_state_means, sigma2_filt,
                transition_functions, transition_covariance
            )
        )

        # reconstruction covariance matrices
        smoothed_state_covariances = np.zeros(sigma2_smooth.shape)
        for t in range(n_timesteps):
            smoothed_state_covariances[t] = (
                sigma2_smooth[t].T.dot(sigma2_smooth[t])
            )

        return (smoothed_state_means, smoothed_state_covariances)
