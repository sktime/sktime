"""Unscented Kalman Filter.

=========================================
Inference for Non-Linear Gaussian Systems
=========================================

This module contains the Unscented Kalman Filter (Wan, van der Merwe 2000)
for state estimation in systems with non-Gaussian noise and non-linear dynamics
"""

from collections import namedtuple

import numpy as np
from numpy import ma
from scipy import linalg

from .standard import _arg_or_default, _determine_dimensionality, _last_dims
from .utils import (
    array1d,
    array2d,
    check_random_state,
    get_params,
    preprocess_arguments,
)
from .utils_numpy import newbyteorder

# represents a collection of sigma points and their associated weights. one
# point per row
SigmaPoints = namedtuple(
    "SigmaPoints", ["points", "weights_mean", "weights_covariance"]
)


# represents mean and covariance of a multivariate normal distribution
Moments = namedtuple("Moments", ["mean", "covariance"])


def points2moments(points, sigma_noise=None):
    """Calculate estimated mean and covariance of sigma points.

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
    """
    (points, weights_mu, weights_sigma) = points
    mu = points.T.dot(weights_mu)
    points_diff = points.T - mu[:, np.newaxis]
    sigma = points_diff.dot(np.diag(weights_sigma)).dot(points_diff.T)
    if sigma_noise is not None:
        sigma = sigma + sigma_noise
    return Moments(mu.ravel(), sigma)


def moments2points(moments, alpha=None, beta=None, kappa=None):
    """Calculate "sigma points" used in Unscented Kalman Filter.

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
    """
    (mu, sigma) = moments
    n_dim = len(mu)
    mu = array2d(mu, dtype=float)

    if alpha is None:
        alpha = 1.0
    if beta is None:
        beta = 0.0
    if kappa is None:
        kappa = 3.0 - n_dim

    # compute sqrt(sigma)
    sigma2 = linalg.cholesky(sigma).T

    # Calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    c = n_dim + lamda

    # calculate the sigma points; that is,
    #   mu
    #   mu + each column of sigma2 * sqrt(c)
    #   mu - each column of sigma2 * sqrt(c)
    # Each column of points is one of these.
    points = np.tile(mu.T, (1, 2 * n_dim + 1))
    points[:, 1 : (n_dim + 1)] += sigma2 * np.sqrt(c)
    points[:, (n_dim + 1) :] -= sigma2 * np.sqrt(c)

    # Calculate weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    return SigmaPoints(points.T, weights_mean, weights_cov)


def unscented_transform(points, f=None, points_noise=None, sigma_noise=None):
    """Apply the Unscented Transform to a set of points.

    Apply f to points (with secondary argument points_noise, if available),
    then approximate the resulting mean and covariance. If sigma_noise is
    available, treat it as additional variance due to additive noise.

    Parameters
    ----------
    points : [n_points, n_dim_state] SigmaPoints
        points to pass into f's first argument and associated weights if f is
        defined. If f is unavailable, then f is assumed to be the identity
        function.
    f : [n_dim_state, n_dim_state_noise] -> [n_dim_state] function
        transition function from time t to time t+1, if available.
    points_noise : [n_points, n_dim_state_noise] array
        points to pass into f's second argument, if any
    sigma_noise : [n_dim_state, n_dim_state] array
        covariance matrix for additive noise, if any

    Returns
    -------
    points_pred : [n_points, n_dim_state] SigmaPoints
        points transformed by f with same weights
    moments_pred : [n_dim_state] Moments
        moments associated with points_pred
    """
    n_points, n_dim_state = points.points.shape
    (points, weights_mean, weights_covariance) = points

    # propagate points through f
    if f is not None:
        if points_noise is None:
            points_pred = [f(points[i]) for i in range(n_points)]
        else:
            points_noise = points_noise.points
            points_pred = [f(points[i], points_noise[i]) for i in range(n_points)]
    else:
        points_pred = points

    # make each row a predicted point
    points_pred = np.vstack(points_pred)
    points_pred = SigmaPoints(points_pred, weights_mean, weights_covariance)

    # calculate approximate mean, covariance
    moments_pred = points2moments(points_pred, sigma_noise)

    return (points_pred, moments_pred)


def unscented_correct(cross_sigma, moments_pred, obs_moments_pred, z):
    """Correct predicted state estimates with an observation.

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
    """
    mu_pred, sigma_pred = moments_pred
    obs_mu_pred, obs_sigma_pred = obs_moments_pred

    if not np.any(ma.getmask(z)):
        # calculate Kalman gain
        K = cross_sigma.dot(linalg.pinv(obs_sigma_pred))

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        sigma_filt = sigma_pred - K.dot(cross_sigma.T)
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma_filt = sigma_pred
    return Moments(mu_filt, sigma_filt)


def augmented_points(momentses):
    """Calculate sigma points for augmented UKF.

    Parameters
    ----------
    momentses : list of Moments
        means and covariances for multiple multivariate normals

    Returns
    -------
    pointses : list of Points
        sigma points for each element of momentses
    """
    # stack everything together
    means, covariances = zip(*momentses)
    mu_aug = np.concatenate(means)
    sigma_aug = linalg.block_diag(*covariances)
    moments_aug = Moments(mu_aug, sigma_aug)

    # turn augmented representation into sigma points
    points_aug = moments2points(moments_aug)

    # unstack everything
    dims = [len(m) for m in means]
    result = []
    start = 0
    for i in range(len(dims)):
        end = start + dims[i]
        part = SigmaPoints(
            points_aug.points[:, start:end],
            points_aug.weights_mean,
            points_aug.weights_covariance,
        )
        result.append(part)
        start = end

    # return
    return result


def augmented_unscented_filter_points(
    mean_state, covariance_state, covariance_transition, covariance_observation
):
    """Extract sigma points using augmented state representation.

    Primarily used as a pre-processing step before predicting and updating in
    the Augmented UKF.

    Parameters
    ----------
    mean_state : [n_dim_state] array
        mean of state at time t given observations from time steps 0...t
    covariance_state : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from time steps 0...t
    covariance_transition : [n_dim_state, n_dim_state] array
        covariance of zero-mean noise resulting from transitioning from time
        step t to t+1
    covariance_observation : [n_dim_obs, n_dim_obs] array
        covariance of zero-mean noise resulting from observation state at time
        t+1

    Returns
    -------
    points_state : [2 * n_dim_state + 1, n_dim_state] SigmaPoints
        sigma points for state at time t
    points_transition : [2 * n_dim_state + 1, n_dim_state] SigmaPoints
        sigma points for transition noise between time t and t+1
    points_observation : [2 * n_dim_state + 1, n_dim_obs] SigmaPoints
        sigma points for observation noise at time step t+1
    """
    # get sizes of dimensions
    n_dim_state = covariance_state.shape[0]
    n_dim_obs = covariance_observation.shape[0]

    # extract sigma points using augmented representation
    state_moments = Moments(mean_state, covariance_state)
    transition_noise_moments = Moments(np.zeros(n_dim_state), covariance_transition)
    observation_noise_moments = Moments(np.zeros(n_dim_obs), covariance_observation)

    (points_state, points_transition, points_observation) = augmented_points(
        [state_moments, transition_noise_moments, observation_noise_moments]
    )
    return (points_state, points_transition, points_observation)


def unscented_filter_predict(
    transition_function, points_state, points_transition=None, sigma_transition=None
):
    """Predict next state distribution.

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
    assert (
        points_transition is not None or sigma_transition is not None
    ), "Your system is noiseless? really?"
    (points_pred, moments_pred) = unscented_transform(
        points_state,
        transition_function,
        points_noise=points_transition,
        sigma_noise=sigma_transition,
    )
    return (points_pred, moments_pred)


def unscented_filter_correct(
    observation_function,
    moments_pred,
    points_pred,
    observation,
    points_observation=None,
    sigma_observation=None,
):
    """Integrate new observation to correct state estimates.

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
    (obs_points_pred, obs_moments_pred) = unscented_transform(
        points_pred,
        observation_function,
        points_noise=points_observation,
        sigma_noise=sigma_observation,
    )

    # Calculate Cov(x_t, z_t | z_{0:t-1})
    sigma_pair = (
        ((points_pred.points - moments_pred.mean).T)
        .dot(np.diag(points_pred.weights_mean))
        .dot(obs_points_pred.points - obs_moments_pred.mean)
    )

    # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
    moments_filt = unscented_correct(
        sigma_pair, moments_pred, obs_moments_pred, observation
    )
    return moments_filt


def augmented_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    """Apply the Unscented Kalman Filter with arbitrary noise.

    Parameters
    ----------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and the
        process noise and outputs the next state.
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
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
    """
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    # TODO use _augumented_unscented_filter_update here
    for t in range(T):
        # Calculate sigma points for augmented state:
        #   [actual state, transition noise, observation noise]
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        # extract sigma points using augmented representation
        (
            points_state,
            points_transition,
            points_observation,
        ) = augmented_unscented_filter_points(mu, sigma, Q, R)

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1}) and sigma points
        # for P(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (points_pred, moments_pred) = unscented_filter_predict(
                transition_function, points_state, points_transition=points_transition
            )

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma_filt[t] = unscented_filter_correct(
            observation_function,
            moments_pred,
            points_pred,
            Z[t],
            points_observation=points_observation,
        )

    return (mu_filt, sigma_filt)


def augmented_unscented_smoother(mu_filt, sigma_filt, f, Q):
    """Apply the Unscented Kalman Smoother with arbitrary noise.

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and the
        process noise and outputs the next state.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of state at time t given observations from times
        [0, T-1]
    sigma_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of state at time t given observations from
        times [0, T-1]
    """
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma_smooth = np.zeros(sigma_filt.shape)
    mu_smooth[-1], sigma_smooth[-1] = mu_filt[-1], sigma_filt[-1]

    for t in reversed(range(T - 1)):
        # get sigma points for [state, transition noise]
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        moments_state = Moments(mu, sigma)
        moments_transition_noise = Moments(np.zeros(n_dim_state), Q)
        (points_state, points_transition) = augmented_points(
            [moments_state, moments_transition_noise]
        )

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = unscented_transform(
            points_state, f_t, points_noise=points_transition
        )

        # Calculate Cov(x_{t+1}, x_t | z_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean)
            .T.dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean)
            .T
        )

        # compute smoothed mean, covariance
        smoother_gain = sigma_pair.dot(linalg.pinv(moments_pred.covariance))
        mu_smooth[t] = mu_filt[t] + smoother_gain.dot(
            mu_smooth[t + 1] - moments_pred.mean
        )
        sigma_smooth[t] = sigma_filt[t] + smoother_gain.dot(
            sigma_smooth[t + 1] - moments_pred.covariance
        ).dot(smoother_gain.T)

    return (mu_smooth, sigma_smooth)


def additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    """Apply the Unscented Kalman Filter with additive noise.

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
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
    """
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        points_state = moments2points(Moments(mu, sigma))

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, moments_pred) = unscented_filter_predict(
                transition_function, points_state, sigma_transition=Q
            )
            points_pred = moments2points(moments_pred)

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma_filt[t] = unscented_filter_correct(
            observation_function, moments_pred, points_pred, Z[t], sigma_observation=R
        )

    return (mu_filt, sigma_filt)


def additive_unscented_smoother(mu_filt, sigma_filt, f, Q):
    """Apply the Unscented Kalman Filter assuming additive noise.

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
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
    sigma_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of state at time t given observations from
        times [0, T-1]
    """
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma_smooth = np.zeros(sigma_filt.shape)
    mu_smooth[-1], sigma_smooth[-1] = mu_filt[-1], sigma_filt[-1]

    for t in reversed(range(T - 1)):
        # get sigma points for state
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        moments_state = Moments(mu, sigma)
        points_state = moments2points(moments_state)

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, moments_pred) = unscented_transform(
            points_state, f_t, sigma_noise=Q
        )

        # Calculate Cov(x_{t+1}, x_t | z_{0:t-1})
        sigma_pair = (
            (points_pred.points - moments_pred.mean)
            .T.dot(np.diag(points_pred.weights_covariance))
            .dot(points_state.points - moments_state.mean)
            .T
        )

        # compute smoothed mean, covariance
        smoother_gain = sigma_pair.dot(linalg.pinv(moments_pred.covariance))
        mu_smooth[t] = mu_filt[t] + smoother_gain.dot(
            mu_smooth[t + 1] - moments_pred.mean
        )
        sigma_smooth[t] = sigma_filt[t] + smoother_gain.dot(
            sigma_smooth[t + 1] - moments_pred.covariance
        ).dot(smoother_gain.T)

    return (mu_smooth, sigma_smooth)


class UnscentedMixin:
    """Methods shared by all Unscented Kalman Filter implementations."""

    def __init__(
        self,
        transition_functions=None,
        observation_functions=None,
        transition_covariance=None,
        observation_covariance=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        n_dim_state=None,
        n_dim_obs=None,
        random_state=None,
    ):
        # determine size of state and observation space
        n_dim_state = _determine_dimensionality(
            [
                (transition_covariance, array2d, -2),
                (initial_state_covariance, array2d, -2),
                (initial_state_mean, array1d, -1),
            ],
            n_dim_state,
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)], n_dim_obs
        )

        # set parameters
        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.random_state = random_state

    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults."""
        arguments = get_params(self)
        defaults = self._default_parameters()
        converters = self._converters()

        processed = preprocess_arguments([arguments, defaults], converters)
        return (
            processed["transition_functions"],
            processed["observation_functions"],
            processed["transition_covariance"],
            processed["observation_covariance"],
            processed["initial_state_mean"],
            processed["initial_state_covariance"],
        )

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format."""
        obs = ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs

    def _converters(self):
        return {
            "transition_functions": array1d,
            "observation_functions": array1d,
            "transition_covariance": array2d,
            "observation_covariance": array2d,
            "initial_state_mean": array1d,
            "initial_state_covariance": array2d,
            "n_dim_state": int,
            "n_dim_obs": int,
            "random_state": check_random_state,
        }


class UnscentedKalmanFilter(UnscentedMixin):
    r"""General (aka Augmented) Unscented Kalman Filter.

    The General Unscented Kalman Filter is governed by the following equations:

    .. math::

        x_0       &\sim \text{Normal}(\mu_0, \Sigma_0)  \\
        x_{t+1}   &=    f_t(x_t, \text{Normal}(0, Q))   \\
        z_{t}     &=    g_t(x_t, \text{Normal}(0, R))

    Notice that although the input noise to the state transition equation and
    the observation equation are both normally distributed, any non-linear
    transformation may be applied afterwards.  This allows for greater
    generality, but at the expense of computational complexity.  The complexity
    of :class:`UnscentedKalmanFilter.filter()` is :math:`O(T(2n+m)^3)`
    where :math:`T` is the number of time steps, :math:`n` is the size of the
    state space, and :math:`m` is the size of the observation space.

    If your noise is simply additive, consider using the
    :class:`AdditiveUnscentedKalmanFilter`

    Parameters
    ----------
    transition_functions : function or [n_timesteps-1] array of functions
        transition_functions[t] is a function of the state and the transition
        noise at time t and produces the state at time t+1.  Also known as
        :math:`f_t`.
    observation_functions : function or [n_timesteps] array of functions
        observation_functions[t] is a function of the state and the observation
        noise at time t and produces the observation at time t.  Also known as
        :math:`g_t`.
    transition_covariance : [n_dim_state, n_dim_state] array
        transition noise covariance matrix. Also known as :math:`Q`.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix. Also known as :math:`R`.
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution. Also known as :math:`\mu_0`
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution. Also known as
        :math:`\Sigma_0`
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_covariance`, or
        `initial_state_mean`, `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_covariance`.
    random_state : optional, int or RandomState
        seed for random sample generation
    """

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        """Sample from model defined by the Unscented Kalman Filter.

        Parameters
        ----------
        n_timesteps : int
            number of time steps
        initial_state : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        random_state : optional, int or Random
            random number generator
        """
        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if initial_state is None:
            initial_state = rng.multivariate_normal(
                initial_state_mean, initial_state_covariance
            )

        # logic for generating samples
        x = np.zeros((n_timesteps, n_dim_state))
        z = np.zeros((n_timesteps, n_dim_obs))
        for t in range(n_timesteps):
            if t == 0:
                x[0] = initial_state
            else:
                transition_func = _last_dims(transition_functions, t - 1, ndims=1)[0]
                cov = newbyteorder(transition_covariance, "=")
                transition_noise = rng.multivariate_normal(np.zeros(n_dim_state), cov)
                x[t] = transition_func(x[t - 1], transition_noise)

            observation_function = _last_dims(observation_functions, t, ndims=1)[0]
            cov = newbyteorder(observation_covariance, "=")
            observation_noise = rng.multivariate_normal(np.zeros(n_dim_obs), cov)
            z[t] = observation_function(x[t], observation_noise)

        return (x, ma.asarray(z))

    def filter(self, Z):
        """Run Unscented Kalman Filter.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, t]
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, t]
        """
        Z = self._parse_observations(Z)

        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (filtered_state_means, filtered_state_covariances) = augmented_unscented_filter(
            initial_state_mean,
            initial_state_covariance,
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            Z,
        )

        return (filtered_state_means, filtered_state_covariances)

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_function=None,
        transition_covariance=None,
        observation_function=None,
        observation_covariance=None,
    ):
        r"""Update a Kalman Filter state estimate.

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
        (
            transition_functions,
            observation_functions,
            transition_cov,
            observation_cov,
            _,
            _,
        ) = self._initialize_parameters()

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
            transition_covariance, transition_cov, 2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov, 2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        # make sigma points
        (
            points_state,
            points_transition,
            points_observation,
        ) = augmented_unscented_filter_points(
            filtered_state_mean,
            filtered_state_covariance,
            transition_covariance,
            observation_covariance,
        )

        # predict
        (points_pred, moments_pred) = unscented_filter_predict(
            transition_function, points_state, points_transition
        )

        # correct
        (
            next_filtered_state_mean,
            next_filtered_state_covariance,
        ) = unscented_filter_correct(
            observation_function,
            moments_pred,
            points_pred,
            observation,
            points_observation=points_observation,
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, Z):
        """Run Unscented Kalman Smoother.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        """
        Z = self._parse_observations(Z)

        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (filtered_state_means, filtered_state_covariances) = self.filter(Z)
        (
            smoothed_state_means,
            smoothed_state_covariances,
        ) = augmented_unscented_smoother(
            filtered_state_means,
            filtered_state_covariances,
            transition_functions,
            transition_covariance,
        )

        return (smoothed_state_means, smoothed_state_covariances)

    def _default_parameters(self):
        return {
            "transition_functions": lambda state, noise: state + noise,
            "observation_functions": lambda state, noise: state + noise,
            "transition_covariance": np.eye(self.n_dim_state),
            "observation_covariance": np.eye(self.n_dim_obs),
            "initial_state_mean": np.zeros(self.n_dim_state),
            "initial_state_covariance": np.eye(self.n_dim_state),
            "random_state": 0,
        }


class AdditiveUnscentedKalmanFilter(UnscentedMixin):
    r"""Unscented Kalman Filter with additive noise.

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
    """

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        """Sample from model defined by the Unscented Kalman Filter.

        Parameters
        ----------
        n_timesteps : int
            number of time steps
        initial_state : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        """
        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if initial_state is None:
            initial_state = rng.multivariate_normal(
                initial_state_mean, initial_state_covariance
            )

        # logic for generating samples
        x = np.zeros((n_timesteps, n_dim_state))
        z = np.zeros((n_timesteps, n_dim_obs))
        for t in range(n_timesteps):
            if t == 0:
                x[0] = initial_state
            else:
                transition_func = _last_dims(transition_functions, t - 1, ndims=1)[0]
                cov = newbyteorder(transition_covariance, "=")
                transition_noise = rng.multivariate_normal(np.zeros(n_dim_state), cov)
                x[t] = transition_func(x[t - 1]) + transition_noise

            observation_function = _last_dims(observation_functions, t, ndims=1)[0]
            cov = newbyteorder(observation_covariance, "=")
            observation_noise = rng.multivariate_normal(np.zeros(n_dim_obs), cov)
            z[t] = observation_function(x[t]) + observation_noise

        return (x, ma.asarray(z))

    def filter(self, Z):
        """Run Unscented Kalman Filter.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, t]
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, t]
        """
        Z = self._parse_observations(Z)

        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (filtered_state_means, filtered_state_covariances) = additive_unscented_filter(
            initial_state_mean,
            initial_state_covariance,
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            Z,
        )

        return (filtered_state_means, filtered_state_covariances)

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_function=None,
        transition_covariance=None,
        observation_function=None,
        observation_covariance=None,
    ):
        r"""Update a Kalman Filter state estimate.

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
        (
            transition_functions,
            observation_functions,
            transition_cov,
            observation_cov,
            _,
            _,
        ) = self._initialize_parameters()

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
            transition_covariance, transition_cov, 2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov, 2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = unscented_filter_predict(
            transition_function, points_state, sigma_transition=transition_covariance
        )
        points_pred = moments2points(moments_pred)

        # correct
        (
            next_filtered_state_mean,
            next_filtered_state_covariance,
        ) = unscented_filter_correct(
            observation_function,
            moments_pred,
            points_pred,
            observation,
            sigma_observation=observation_covariance,
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, Z):
        """Run Unscented Kalman Smoother.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        """
        Z = ma.asarray(Z)

        (
            transition_functions,
            observation_functions,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (filtered_state_means, filtered_state_covariances) = self.filter(Z)
        (
            smoothed_state_means,
            smoothed_state_covariances,
        ) = additive_unscented_smoother(
            filtered_state_means,
            filtered_state_covariances,
            transition_functions,
            transition_covariance,
        )

        return (smoothed_state_means, smoothed_state_covariances)

    def _default_parameters(self):
        return {
            "transition_functions": lambda state: state,
            "observation_functions": lambda state: state,
            "transition_covariance": np.eye(self.n_dim_state),
            "observation_covariance": np.eye(self.n_dim_obs),
            "initial_state_mean": np.zeros(self.n_dim_state),
            "initial_state_covariance": np.eye(self.n_dim_state),
            "random_state": 0,
        }
