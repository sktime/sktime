"""Bierman's version of the Kalman Filter.

=====================================
Inference for Linear-Gaussian Systems
=====================================

This module implements Bierman's version of the Kalman Filter.  In particular,
the UDU' decomposition of the covariance matrix is used instead of the full
matrix, where U is upper triangular and D is diagonal.
"""

import warnings

import numpy as np
from scipy import linalg

from ..standard import (
    DIM,
    KalmanFilter,
    _arg_or_default,
    _em,
    _last_dims,
    _loglikelihoods,
    _smooth,
    _smooth_pair,
)
from ..utils import get_params


def _reconstruct_covariances(covariances):
    """Reconstruct covariance matrices given their UDU' factorizations."""
    if isinstance(covariances, UDU_decomposition):
        covariances = np.asarray([covariances])

    n_timesteps = covariances.shape[0]
    n_dim_state = covariances[0].U.shape[0]
    result = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    for t in range(n_timesteps):
        result[t] = covariances[t].reconstruct()

    return result


class UDU_decomposition:
    """Represents a UDU' decomposition of a matrix."""

    def __init__(self, U, D):
        self.U = U
        self.D = D

    def reconstruct(self):
        """Reconstruct the original matrix from the UDU' decomposition."""
        return self.U.dot(np.diag(self.D)).dot(self.U.T)


def udu(M):
    """Construct the UDU' decomposition of a positive, semidefinite matrix M.

    Parameters
    ----------
    M : [n, n] array
        Matrix to factorize

    Returns
    -------
    UDU : UDU_decomposition of size n
        UDU' representation of M
    """
    assert np.allclose(M, M.T), "M must be symmetric, positive semidefinite"
    n = M.shape[0]

    # perform Bierman's COV2UD subroutine (fucking inclusive indices)
    M = np.triu(M)
    U = np.eye(n)
    d = np.zeros(n)
    for j in reversed(range(2, n + 1)):
        d[j - 1] = M[j - 1, j - 1]
        if d[j - 1] > 0:
            alpha = 1.0 / d[j - 1]
        else:
            alpha = 0.0
        for k in range(1, j):
            beta = M[k - 1, j - 1]
            U[k - 1, j - 1] = alpha * beta
            M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]

    d[0] = M[0, 0]

    return UDU_decomposition(U, d)


def decorrelate_observations(
    observation_matrices, observation_offsets, observation_covariance, observations
):
    """Make each coordinate of all observation independent.

    Modify observations and all associated parameters such that all observation
    indices are expected to be independent.

    Parameters
    ----------
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs, \
    n_dim_obs] array
        observation matrix
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array
        observations for times [0...n_timesteps-1]
    observation_covariance : [n_timesteps, n_dim_obs, n_dim_obs] or \
    [n_dim_obs, n_dim_obs] array
        observation covariance matrix
    observations : [n_timesteps, n_dim_obs] array
        observations from times [0...n_timesteps-1].  If `observations` is a
        masked array and any of `observations[t]` is masked, then
        `observations[t]` will be treated as a missing observation.

    Returns
    -------
    observation_matrices2 : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs, \
    n_dim_obs] array
        observation matrix with each index decorrelated
    observation_offsets2 : [n_timesteps, n_dim_obs] or [n_dim_obs] array
        observations for times [0...n_timesteps-1] with each index decorrelated
    observation_covariance : [n_timesteps, n_dim_obs, n_dim_obs] or \
    [n_dim_obs, n_dim_obs] array
        observation covariance matrix with each index decorrelated
    observations2 : [n_timesteps, n_dim_obs] array
        observations from times [0...n_timesteps-1] with each index
        decorrelated.
    """
    n_dim_obs = observations.shape[-1]

    # calculate (R^{1/2})^{-1}
    observation_covariance2 = linalg.cholesky(observation_covariance, lower=True)
    observation_covariance_inv = linalg.pinv(observation_covariance2)

    # decorrelate observation_matrices
    observation_matrices2 = np.copy(observation_matrices)
    if len(observation_matrices.shape) == DIM["observation_matrices"] + 1:
        n_timesteps = observation_matrices.shape[0]
        for t in range(n_timesteps):
            observation_matrices2[t] = observation_covariance_inv.dot(
                observation_matrices[t]
            )
    else:
        observation_matrices2 = observation_covariance_inv.dot(observation_matrices)

    # decorrelate observation_offsets
    observation_offsets2 = observation_covariance_inv.dot(observation_offsets.T).T

    # decorrelate observations
    observations2 = observation_covariance_inv.dot(observations.T).T

    return (
        observation_matrices2,
        observation_offsets2,
        np.eye(n_dim_obs),
        observations2,
    )


def _filter_predict(
    transition_matrix,
    transition_covariance,
    transition_offset,
    current_state_mean,
    current_state_covariance,
):
    r"""Calculate the mean and covariance of :math:`P(x_{t+1} | z_{0:t})`.

    Using the mean and covariance of :math:`P(x_t | z_{0:t})`, calculate the
    mean and covariance of :math:`P(x_{t+1} | z_{0:t})`.

    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state} array
        state transition matrix from time t to t+1
    transition_covariance : [n_dim_state, n_dim_state] array
        covariance matrix for state transition from time t to t+1
    transition_offset : [n_dim_state] array
        offset for state transition from time t to t+1
    current_state_mean: [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    current_state_covariance: n_dim_state UDU_decomposition
        UDU' decomposition of the covariance of state at time t given
        observations from times [0...t]

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of state at time t+1 given observations from times [0...t]
    predicted_state_covariance : n_dim_state UDU_decomposition
        UDU' decomposition of the covariance of state at time t+1 given
        observations from times [0...t]

    References
    ----------
    * Gibbs, Bruce P. Advanced Kalman Filtering, Least-Squares, and Modeling: A
      Practical Handbook. Page 401.
    """
    # predict new mean
    predicted_state_mean = (
        np.dot(transition_matrix, current_state_mean) + transition_offset
    )

    # predict new covariance
    predicted_state_covariance = udu(
        transition_matrix.dot(current_state_covariance.reconstruct()).dot(
            transition_matrix.T
        )
        + transition_covariance
    )
    return (predicted_state_mean, predicted_state_covariance)


def _filter_correct_single(UDU, h, R):
    """Correct predicted state covariance, calculate one column of the Kalman gain.

    Parameters
    ----------
    UDU : [n_dim_state, n_dim_state] array
        UDU' decomposition of the covariance matrix for state at time t given
        observations from time 0...t-1 and the first i-1 observations at time t
    h : [n_dim_state] array
        i-th row of observation matrix
    R : float
        covariance corresponding to the i-th coordinate of the observation

    Returns
    -------
    corrected_state_covariance : n_dim_state UDU_decomposition
        UDU' decomposition of the covariance matrix for state at time t given
        observations from time 0...t-1 and the first i observations at time t
    k : [n_dim_state] array
        Kalman gain for i-th coordinate of the observation at time t

    References
    ----------
    * Gibbs, Bruce P. Advanced Kalman Filtering, Least-Squares, and Modeling: A
      Practical Handbook. Page 396
    """
    n_dim_state = len(h)

    U = UDU.U
    D = UDU.D
    f = h.dot(U)  # len(f) == n_dim_state
    g = np.diag(D).dot(f)  # len(g) == n_dim-state
    alpha = f.dot(g) + R

    gamma = np.zeros(n_dim_state)
    U_bar = np.zeros((n_dim_state, n_dim_state))
    D_bar = np.zeros(n_dim_state)
    k = np.zeros(n_dim_state)

    gamma[0] = R + g[0] * f[0]
    D_bar[0] = D[0] * R / gamma[0]
    k[0] = g[0]

    U_bar[0, 0] = 1
    for j in range(1, n_dim_state):
        gamma[j] = gamma[j - 1] + g[j] * f[j]
        D_bar[j] = D[j] * gamma[j - 1] / gamma[j]
        U_bar[:, j] = U[:, j] - (f[j] / gamma[j - 1]) * k
        k = k + g[j] * U[:, j]

    return (UDU_decomposition(U_bar, D_bar), k / alpha)


def _filter_correct(
    observation_matrix,
    observation_covariance,
    observation_offset,
    predicted_state_mean,
    predicted_state_covariance,
    observation,
):
    r"""Correct a predicted state with a Kalman Filter update.

    Incorporate observation `observation` from time `t` to turn
    :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

    Parameters
    ----------
    observation_matrix : [n_dim_obs, n_dim_state] array
        observation matrix for time t
    observation_covariance : n_dim_state UDU_decomposition
        UDU' decomposition of observation covariance matrix for observation at
        time t
    observation_offset : [n_dim_obs] array
        offset for observation at time t
    predicted_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t-1]
    predicted_state_covariance : n_dim_state UDU_decomposition
        UDU' decomposition of the covariance of state at time t given
        observations from times [0...t-1]
    observation : [n_dim_obs] array
        observation at time t.  If `observation` is a masked array and any of
        its values are masked, the observation will be ignored.

    Returns
    -------
    corrected_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    corrected_state_covariance : n_dim_state UDU_decomposition
        UDU' decomposition of the covariance of state at time t given
        observations from times [0...t]

    References
    ----------
    * Gibbs, Bruce P. Advanced Kalman Filtering, Least-Squares, and Modeling: A
      Practical Handbook. Page 394-396
    """
    if not np.any(np.ma.getmask(observation)):
        # extract size of state space
        n_dim_obs = len(observation)

        # calculate corrected state mean, covariance
        corrected_state_mean = predicted_state_mean
        corrected_state_covariance = predicted_state_covariance
        for i in range(n_dim_obs):
            # extract components for updating i-th coordinate's covariance
            o = observation[i]
            b = observation_offset[i]
            h = observation_matrix[i]
            R = observation_covariance[i, i]

            # calculate new UDU' decomposition for corrected_state_covariance
            # and a new column of the kalman gain
            (corrected_state_covariance, k) = _filter_correct_single(
                corrected_state_covariance, h, R
            )

            # update corrected state mean
            predicted_observation_mean = h.dot(corrected_state_mean) + b
            corrected_state_mean = corrected_state_mean + k.dot(
                o - predicted_observation_mean
            )

    else:
        n_dim_obs = len(observation)

        corrected_state_mean = predicted_state_mean
        corrected_state_covariance = predicted_state_covariance

    return (corrected_state_mean, corrected_state_covariance)


def _filter(
    transition_matrices,
    observation_matrices,
    transition_covariance,
    observation_covariance,
    transition_offsets,
    observation_offsets,
    initial_state_mean,
    initial_state_covariance,
    observations,
):
    """Apply the Kalman Filter.

    Calculate posterior distribution over hidden states given observations up
    to and including the current time step.

    Parameters
    ----------
    transition_matrices : [n_timesteps-1,n_dim_state,n_dim_state] or \
    [n_dim_state,n_dim_state] array
        state transition matrices
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs, \
    n_dim_obs] array
        observation matrix
    transition_covariance : [n_dim_state, n_dim_state] array
        state transition covariance matrix
    observation_covariance : [n_timesteps, n_dim_obs, n_dim_obs] or \
    [n_dim_obs, n_dim_obs] array
        observation covariance matrix
    transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
    array
        state offset
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array
        observations for times [0...n_timesteps-1]
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    observations : [n_timesteps, n_dim_obs] array
        observations from times [0...n_timesteps-1].  If `observations` is a
        masked array and any of `observations[t]` is masked, then
        `observations[t]` will be treated as a missing observation.

    Returns
    -------
    predicted_state_means : [n_timesteps, n_dim_state] array
        `predicted_state_means[t]` = mean of hidden state at time t given
        observations from times [0...t-1]
    predicted_state_covariances : [n_timesteps] array of n_dim_state \
    UDU_decompositions
        `predicted_state_covariances[t]` = UDU' decomposition of the covariance
        of hidden state at time t given observations from times [0...t-1]
    filtered_state_means : [n_timesteps, n_dim_state] array
        `filtered_state_means[t]` = mean of hidden state at time t given
        observations from times [0...t]
    filtered_state_covariances : [n_timesteps] array of n_dim_state \
    UDU_decompositions
        `filtered_state_covariances[t]` = UDU' decomposition of the covariance
        of hidden state at time t given observations from times [0...t]
    """
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)

    # construct containers for outputs
    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros(n_timesteps, dtype=object)
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros(n_timesteps, dtype=object)

    # initialize filter
    initial_state_covariance = udu(initial_state_covariance)
    (
        observation_matrices,
        observation_offsets,
        observation_covariance,
        observations,
    ) = decorrelate_observations(
        observation_matrices, observation_offsets, observation_covariance, observations
    )

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance
        else:
            transition_matrix = _last_dims(transition_matrices, t - 1)
            transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
            predicted_state_means[t], predicted_state_covariances[t] = _filter_predict(
                transition_matrix,
                transition_covariance,
                transition_offset,
                filtered_state_means[t - 1],
                filtered_state_covariances[t - 1],
            )

        observation_matrix = _last_dims(observation_matrices, t)
        observation_offset = _last_dims(observation_offsets, t, ndims=1)
        (filtered_state_means[t], filtered_state_covariances[t]) = _filter_correct(
            observation_matrix,
            observation_covariance,
            observation_offset,
            predicted_state_means[t],
            predicted_state_covariances[t],
            observations[t],
        )

    return (
        predicted_state_means,
        predicted_state_covariances,
        filtered_state_means,
        filtered_state_covariances,
    )


class BiermanKalmanFilter(KalmanFilter):
    r"""Kalman Filter based on UDU' decomposition.

    Parameters
    ----------
    transition_matrices : [n_timesteps-1, n_dim_state, n_dim_state] or \
    [n_dim_state,n_dim_state] array-like
        Also known as :math:`A`.  state transition matrix between times t and
        t+1 for t in [0...n_timesteps-2]
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs, \
    n_dim_obs] array-like
        Also known as :math:`C`.  observation matrix for times
        [0...n_timesteps-1]
    transition_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`Q`.  state transition covariance matrix for times
        [0...n_timesteps-2]
    observation_covariance : [n_dim_obs, n_dim_obs] array-like
        Also known as :math:`R`.  observation covariance matrix for times
        [0...n_timesteps-1]
    transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
    array-like
        Also known as :math:`b`.  state offsets for times [0...n_timesteps-2]
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array-like
        Also known as :math:`d`.  observation offset for times
        [0...n_timesteps-1]
    initial_state_mean : [n_dim_state] array-like
        Also known as :math:`\\mu_0`. mean of initial state distribution
    initial_state_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`\\Sigma_0`.  covariance of initial state
        distribution
    random_state : optional, numpy random state
        random number generator used in sampling
    em_vars : optional, subset of ['transition_matrices', \
    'observation_matrices', 'transition_offsets', 'observation_offsets', \
    'transition_covariance', 'observation_covariance', 'initial_state_mean', \
    'initial_state_covariance'] or 'all'
        if `em_vars` is an iterable of strings only variables in `em_vars`
        will be estimated using EM.  if `em_vars` == 'all', then all
        variables will be estimated.
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_matrices`, `transition_offsets`,
        `transition_covariance`, `initial_state_mean`, or
        `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_matrices`,
        `observation_offsets`, or `observation_covariance`.
    """

    def filter(self, X):
        r"""Apply the Kalman Filter.

        Apply the Kalman Filter to estimate the hidden state at time :math:`t`
        for :math:`t = [0...n_{\\text{timesteps}}-1]` given observations up to
        and including time `t`.  Observations are assumed to correspond to
        times :math:`[0...n_{\\text{timesteps}}-1]`.  The output of this method
        corresponding to time :math:`n_{\\text{timesteps}}-1` can be used in
        :func:`KalmanFilter.filter_update` for online updating.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given observations up to and including the current time step
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] \
        array
            covariance matrix of hidden state distributions for times
            [0...n_timesteps-1] given observations up to and including the
            current time step
        """
        Z = self._parse_observations(X)

        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (_, _, filtered_state_means, filtered_state_covariances) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        filtered_state_covariances = _reconstruct_covariances(
            filtered_state_covariances
        )

        return (filtered_state_means, filtered_state_covariances)

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_matrix=None,
        transition_offset=None,
        transition_covariance=None,
        observation_matrix=None,
        observation_offset=None,
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
        transition_matrix : optional, [n_dim_state, n_dim_state] array
            state transition matrix from time t to t+1.  If unspecified,
            `self.transition_matrices` will be used.
        transition_offset : optional, [n_dim_state] array
            state offset for transition from time t to t+1.  If unspecified,
            `self.transition_offset` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_matrix : optional, [n_dim_obs, n_dim_state] array
            observation matrix at time t+1.  If unspecified,
            `self.observation_matrices` will be used.
        observation_offset : optional, [n_dim_obs] array
            observation offset at time t+1.  If unspecified,
            `self.observation_offset` will be used.
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
        # initialize matrices
        (
            transition_matrices,
            transition_offsets,
            transition_cov,
            observation_matrices,
            observation_offsets,
            observation_cov,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets, 1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets, 1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices, 2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices, 2, "observation_matrix"
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

        # transform filtered_state_covariance into its UDU decomposition
        filtered_state_covariance = udu(filtered_state_covariance)

        # decorrelate observations
        (
            observation_matrix,
            observation_offset,
            observation_covariance,
            observation,
        ) = decorrelate_observations(
            observation_matrix, observation_offset, observation_covariance, observation
        )

        # predict
        predicted_state_mean, predicted_state_covariance = _filter_predict(
            transition_matrix,
            transition_covariance,
            transition_offset,
            filtered_state_mean,
            filtered_state_covariance,
        )

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance) = _filter_correct(
            observation_matrix,
            observation_covariance,
            observation_offset,
            predicted_state_mean,
            predicted_state_covariance,
            observation,
        )

        # reconstruct actual covariance
        next_filtered_state_covariance = _reconstruct_covariances(
            next_filtered_state_covariance
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, X):
        r"""Apply the Kalman Smoother.

        Apply the Kalman Smoother to estimate the hidden state at time
        :math:`t` for :math:`t = [0...n_{\\text{timesteps}}-1]` given all
        observations.  See :func:`_smooth` for more complex output

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given all observations
        smoothed_state_covariances : [n_timesteps, n_dim_state]
            covariances of hidden state distributions for times
            [0...n_timesteps-1] given all observations
        """
        Z = self._parse_observations(X)

        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        # run filter
        (
            predicted_state_means,
            predicted_state_covariances,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        # construct actual covariance matrices
        predicted_state_covariances = _reconstruct_covariances(
            predicted_state_covariances
        )
        filtered_state_covariances = _reconstruct_covariances(
            filtered_state_covariances
        )

        (smoothed_state_means, smoothed_state_covariances) = _smooth(
            transition_matrices,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances,
        )[:2]
        return (smoothed_state_means, smoothed_state_covariances)

    def em(self, X, y=None, n_iter=10, em_vars=None):
        """Apply the EM algorithm.

        Apply the EM algorithm to estimate all parameters specified by
        `em_vars`.  Note that all variables estimated are assumed to be
        constant for all time.  See :func:`_em` for details.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]`'s components is masked, then
            `X[t]` will be treated as a missing observation.
        n_iter : int, optional
            number of EM iterations to perform
        em_vars : iterable of strings or 'all'
            variables to perform EM over.  Any variable not appearing here is
            left untouched.
        """
        Z = self._parse_observations(X)

        # initialize parameters
        (
            self.transition_matrices,
            self.transition_offsets,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_offsets,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        # Create dictionary of variables not to perform EM on
        if em_vars is None:
            em_vars = self.em_vars

        if em_vars == "all":
            given = {}
        else:
            given = {
                "transition_matrices": self.transition_matrices,
                "observation_matrices": self.observation_matrices,
                "transition_offsets": self.transition_offsets,
                "observation_offsets": self.observation_offsets,
                "transition_covariance": self.transition_covariance,
                "observation_covariance": self.observation_covariance,
                "initial_state_mean": self.initial_state_mean,
                "initial_state_covariance": self.initial_state_covariance,
            }
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # If a parameter is time varying, print a warning
        for k, v in get_params(self).items():
            if k in DIM and (k not in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    "{0} has {1} dimensions now; after fitting,"
                    + " it will have dimension {2}"
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str, stacklevel=2)

        # Actual EM iterations
        for _ in range(n_iter):
            # run filter
            (
                predicted_state_means,
                predicted_state_covariances,
                filtered_state_means,
                filtered_state_covariances,
            ) = _filter(
                self.transition_matrices,
                self.observation_matrices,
                self.transition_covariance,
                self.observation_covariance,
                self.transition_offsets,
                self.observation_offsets,
                self.initial_state_mean,
                self.initial_state_covariance,
                Z,
            )

            # reconstruct covariances
            filtered_state_covariances = _reconstruct_covariances(
                filtered_state_covariances
            )
            predicted_state_covariances = _reconstruct_covariances(
                predicted_state_covariances
            )

            # run smoother
            (
                smoothed_state_means,
                smoothed_state_covariances,
                kalman_smoothing_gains,
            ) = _smooth(
                self.transition_matrices,
                filtered_state_means,
                filtered_state_covariances,
                predicted_state_means,
                predicted_state_covariances,
            )

            # calculate pairwise covariances
            sigma_pair_smooth = _smooth_pair(
                smoothed_state_covariances, kalman_smoothing_gains
            )
            (
                self.transition_matrices,
                self.observation_matrices,
                self.transition_offsets,
                self.observation_offsets,
                self.transition_covariance,
                self.observation_covariance,
                self.initial_state_mean,
                self.initial_state_covariance,
            ) = _em(
                Z,
                self.transition_offsets,
                self.observation_offsets,
                smoothed_state_means,
                smoothed_state_covariances,
                sigma_pair_smooth,
                given=given,
            )
        return self

    def loglikelihood(self, X):
        """Calculate the log-likelihood of all observations.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array
            observations for time steps [0...n_timesteps-1]

        Returns
        -------
        likelihood : float
            likelihood of all observations
        """
        Z = self._parse_observations(X)

        # initialize parameters
        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        # apply the Kalman Filter
        (
            predicted_state_means,
            predicted_state_covariances,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        # get likelihoods for each time step
        predicted_state_covariances = _reconstruct_covariances(
            predicted_state_covariances
        )
        loglikelihoods = _loglikelihoods(
            observation_matrices,
            observation_offsets,
            observation_covariance,
            predicted_state_means,
            predicted_state_covariances,
            Z,
        )

        return np.sum(loglikelihoods)
