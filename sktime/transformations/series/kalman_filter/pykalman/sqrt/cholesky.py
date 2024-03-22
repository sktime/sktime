"""
=====================================
Inference for Linear-Gaussian Systems
=====================================

This module implements the Kalman Filter in "Square Root" form (Cholesky
factorization).
"""
import warnings

import numpy as np
from scipy import linalg

from ..standard import _arg_or_default, _determine_dimensionality, \
    _last_dims, _loglikelihoods, _smooth, _smooth_pair, _em, KalmanFilter, DIM
from ..utils import array1d, array2d, check_random_state, \
    get_params


def _reconstruct_covariances(covariance2s):
    '''Reconstruct covariance matrices given their cholesky factors'''
    if len(covariance2s.shape) == 2:
        covariance2s = covariance2s[np.newaxis, :, :]

    T = covariance2s.shape[0]
    covariances = np.zeros(covariance2s.shape)

    for t in range(T):
        M = covariance2s[t]
        covariances[t] = M.dot(M.T)

    return covariances


def _filter_predict(transition_matrix, transition_covariance2,
                    transition_offset, current_state_mean,
                    current_state_covariance2):
    r"""Calculate the mean and covariance of :math:`P(x_{t+1} | z_{0:t})`

    Using the mean and covariance of :math:`P(x_t | z_{0:t})`, calculate the
    mean and covariance of :math:`P(x_{t+1} | z_{0:t})`.

    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state} array
        state transition matrix from time t to t+1
    transition_covariance2 : [n_dim_state, n_dim_state] array
        square root of the covariance matrix for state transition from time
        t to t+1
    transition_offset : [n_dim_state] array
        offset for state transition from time t to t+1
    current_state_mean: [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    current_state_covariance2: [n_dim_state, n_dim_state] array
        square root of the covariance of state at time t given observations
        from times [0...t]

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of state at time t+1 given observations from times [0...t]
    predicted_state_covariance2 : [n_dim_state, n_dim_state] array
        square root of the covariance of state at time t+1 given observations
        from times [0...t]

    References
    ----------
    * Kaminski, Paul G. Square Root Filtering and Smoothing for Discrete
      Processes. July 1971. Page 41.
    """
    n_dim_state = len(current_state_mean)

    # predict new mean
    # x_{t+1|t} = A x_t + b_t
    predicted_state_mean = (
        np.dot(transition_matrix, current_state_mean)
        + transition_offset
    )

    # predict new covariance
    # [S_{k|k-1}^T; 0] = T_1 [ S_{k-1|k-1}^T A^T; Q^{1/2}^T ] for orthonormal T_1
    T, predicted_state_covariance2 = (
        linalg.qr(np.hstack([
            np.dot(transition_matrix, current_state_covariance2),
            transition_covariance2
        ]).T)
    )
    predicted_state_covariance2 = (
        predicted_state_covariance2[:n_dim_state, :n_dim_state].T
    )

    return (predicted_state_mean, predicted_state_covariance2)


def _filter_correct(observation_matrix, observation_covariance2,
                    observation_offset, predicted_state_mean,
                    predicted_state_covariance2, observation):
    r"""Correct a predicted state with a Kalman Filter update

    Incorporate observation `observation` from time `t` to turn
    :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

    Parameters
    ----------
    observation_matrix : [n_dim_obs, n_dim_state] array
        observation matrix for time t
    observation_covariance2 : [n_dim_obs, n_dim_obs] array
        square root of the covariance matrix for observation at time t
    observation_offset : [n_dim_obs] array
        offset for observation at time t
    predicted_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t-1]
    predicted_state_covariance2 : [n_dim_state, n_dim_state] array
        square root of the covariance of state at time t given observations
        from times [0...t-1]
    observation : [n_dim_obs] array
        observation at time t.  If `observation` is a masked array and any of
        its values are masked, the observation will be ignored.

    Returns
    -------
    corrected_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    corrected_state_covariance2 : [n_dim_state, n_dim_state] array
        square root of the covariance of state at time t given observations
        from times [0...t]

    References
    ----------
    * Salzmann, M. A. Some Aspects of Kalman Filtering. August 1988. Page 31.
    """
    if not np.any(np.ma.getmask(observation)):
        # extract size of state space
        n_dim_state = len(predicted_state_mean)
        n_dim_obs = len(observation)

        # construct matrix M = [    R^{1/2}^{T},            0;
        #                       (C S_{t|t-1})^T,  S_{t|t-1}^T]
        M = np.zeros(2 * [n_dim_obs + n_dim_state])
        M[0:n_dim_obs, 0:n_dim_obs] = observation_covariance2.T
        M[n_dim_obs:, 0:n_dim_obs] = observation_matrix.dot(predicted_state_covariance2).T
        M[n_dim_obs:, n_dim_obs:] = predicted_state_covariance2.T

        # solve for [((C P_{t|t-1} C^T + R)^{1/2})^T,         K^T;
        #                                          0,   S_{t|t}^T] = QR(M)
        (_, S) = linalg.qr(M)
        kalman_gain = S[0:n_dim_obs,  n_dim_obs:].T
        N = S[0:n_dim_obs, 0:n_dim_obs].T

        # correct mean
        predicted_observation_mean = (
            np.dot(observation_matrix,
                   predicted_state_mean)
            + observation_offset
        )
        corrected_state_mean = (
            predicted_state_mean
            + np.dot(kalman_gain,
                     np.dot(linalg.pinv(N),
                            observation - predicted_observation_mean)
              )
        )

        corrected_state_covariance2 = S[n_dim_obs:, n_dim_obs:].T
    else:
        n_dim_state = predicted_state_covariance2.shape[0]
        n_dim_obs = observation_matrix.shape[0]
        kalman_gain = np.zeros((n_dim_state, n_dim_obs))

        corrected_state_mean = predicted_state_mean
        corrected_state_covariance2 = predicted_state_covariance2

    return (corrected_state_mean, corrected_state_covariance2)


def _filter(transition_matrices, observation_matrices, transition_covariance,
            observation_covariance, transition_offsets, observation_offsets,
            initial_state_mean, initial_state_covariance, observations):
    """Apply the Kalman Filter

    Calculate posterior distribution over hidden states given observations up
    to and including the current time step.

    Parameters
    ----------
    transition_matrices : [n_timesteps-1,n_dim_state,n_dim_state] or
    [n_dim_state,n_dim_state] array-like
        state transition matrices
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs, \
    n_dim_obs] array-like
        observation matrix
    transition_covariance : [n_timesteps-1,n_dim_state,n_dim_state] or
    [n_dim_state,n_dim_state] array-like
        state transition covariance matrix
    observation_covariance : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs,
    n_dim_obs] array-like
        observation covariance matrix
    transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
    array-like
        state offset
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array-like
        observations for times [0...n_timesteps-1]
    initial_state_mean : [n_dim_state] array-like
        mean of initial state distribution
    initial_state_covariance : [n_dim_state, n_dim_state] array-like
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
    predicted_state_covariance2s : [n_timesteps, n_dim_state, n_dim_state] array
        `predicted_state_covariance2s[t]` = lower triangular factorization of
        the covariance of hidden state at time t given observations from times
        [0...t-1]
    filtered_state_means : [n_timesteps, n_dim_state] array
        `filtered_state_means[t]` = mean of hidden state at time t given
        observations from times [0...t]
    filtered_state_covariance2s : [n_timesteps, n_dim_state] array
        `filtered_state_covariance2s[t]` = lower triangular factorization of
        the covariance of hidden state at time t given observations from times
        [0...t]
    """
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)
    n_dim_obs = observations.shape[1]

    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariance2s = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariance2s = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )
    transition_covariance2 = linalg.cholesky(transition_covariance, lower=True)
    observation_covariance2 = linalg.cholesky(observation_covariance, lower=True)
    initial_state_covariance2 = linalg.cholesky(initial_state_covariance, lower=True)

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariance2s[t] = initial_state_covariance2
        else:
            transition_matrix = _last_dims(transition_matrices, t - 1)
            transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
            predicted_state_means[t], predicted_state_covariance2s[t] = (
                _filter_predict(
                    transition_matrix,
                    transition_covariance2,
                    transition_offset,
                    filtered_state_means[t - 1],
                    filtered_state_covariance2s[t - 1]
                )
            )

        observation_matrix = _last_dims(observation_matrices, t)
        observation_offset = _last_dims(observation_offsets, t, ndims=1)
        (filtered_state_means[t], filtered_state_covariance2s[t]) = (
            _filter_correct(
                observation_matrix,
                observation_covariance2,
                observation_offset,
                predicted_state_means[t],
                predicted_state_covariance2s[t],
                observations[t]
            )
        )

    return (predicted_state_means, predicted_state_covariance2s,
            filtered_state_means, filtered_state_covariance2s)


class CholeskyKalmanFilter(KalmanFilter):
    """Kalman Filter based on Cholesky decomposition

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
        """Apply the Kalman Filter

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

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (_, _, filtered_state_means,
         filtered_state_covariance2s) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance,
                Z
            )
        )

        filtered_state_covariances = (
            _reconstruct_covariances(filtered_state_covariance2s)
        )

        return (filtered_state_means, filtered_state_covariances)

    def filter_update(self, filtered_state_mean, filtered_state_covariance,
                      observation=None, transition_matrix=None,
                      transition_offset=None, transition_covariance=None,
                      observation_matrix=None, observation_offset=None,
                      observation_covariance=None):
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
        (transition_matrices, transition_offsets, transition_cov,
         observation_matrices, observation_offsets, observation_cov,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets,
            1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets,
            1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices,
            2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices,
            2, "observation_matrix"
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

        # turn covariance into cholesky factorizations
        transition_covariance2 = linalg.cholesky(transition_covariance, lower=True)
        observation_covariance2 = linalg.cholesky(observation_covariance, lower=True)
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance, lower=True)

        # predict
        predicted_state_mean, predicted_state_covariance2 = (
            _filter_predict(
                transition_matrix, transition_covariance2,
                transition_offset, filtered_state_mean,
                filtered_state_covariance2
            )
        )

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance2) = (
            _filter_correct(
                observation_matrix, observation_covariance2,
                observation_offset, predicted_state_mean,
                predicted_state_covariance2, observation
            )
        )

        # reconstruct actual covariance
        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_covariance2)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, X):
        """Apply the Kalman Smoother

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

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        # run filter
        (predicted_state_means, predicted_state_covariance2s,
         filtered_state_means, filtered_state_covariance2s) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance, Z
            )
        )

        # construct actual covariance matrices
        predicted_state_covariances = (
            _reconstruct_covariances(predicted_state_covariance2s)
        )
        filtered_state_covariances = (
            _reconstruct_covariances(filtered_state_covariance2s)
        )

        (smoothed_state_means, smoothed_state_covariances) = (
            _smooth(
                transition_matrices, filtered_state_means,
                filtered_state_covariances, predicted_state_means,
                predicted_state_covariances
            )[:2]
        )
        return (smoothed_state_means, smoothed_state_covariances)

    def em(self, X, y=None, n_iter=10, em_vars=None):
        """Apply the EM algorithm

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
        (self.transition_matrices, self.transition_offsets,
         self.transition_covariance, self.observation_matrices,
         self.observation_offsets, self.observation_covariance,
         self.initial_state_mean, self.initial_state_covariance) = (
            self._initialize_parameters()
        )

        # Create dictionary of variables not to perform EM on
        if em_vars is None:
            em_vars = self.em_vars

        if em_vars == 'all':
            given = {}
        else:
            given = {
                'transition_matrices': self.transition_matrices,
                'observation_matrices': self.observation_matrices,
                'transition_offsets': self.transition_offsets,
                'observation_offsets': self.observation_offsets,
                'transition_covariance': self.transition_covariance,
                'observation_covariance': self.observation_covariance,
                'initial_state_mean': self.initial_state_mean,
                'initial_state_covariance': self.initial_state_covariance
            }
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # If a parameter is time varying, print a warning
        for (k, v) in get_params(self).items():
            if k in DIM and (not k in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    '{0} has {1} dimensions now; after fitting, '
                    + 'it will have dimension {2}'
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str)

        # Actual EM iterations
        for i in range(n_iter):
            # run filter
            (predicted_state_means, predicted_state_covariance2s,
             filtered_state_means, filtered_state_covariance2s) = (
                _filter(
                    self.transition_matrices, self.observation_matrices,
                    self.transition_covariance, self.observation_covariance,
                    self.transition_offsets, self.observation_offsets,
                    self.initial_state_mean, self.initial_state_covariance,
                    Z
                )
            )

            # reconstruct covariances
            filtered_state_covariances = (
                _reconstruct_covariances(filtered_state_covariance2s)
            )
            predicted_state_covariances = (
                _reconstruct_covariances(predicted_state_covariance2s)
            )

            # run smoother
            (smoothed_state_means, smoothed_state_covariances,
             kalman_smoothing_gains) = (
                _smooth(
                    self.transition_matrices, filtered_state_means,
                    filtered_state_covariances, predicted_state_means,
                    predicted_state_covariances
                )
            )

            # calculate pairwise covariances
            sigma_pair_smooth = _smooth_pair(
                smoothed_state_covariances,
                kalman_smoothing_gains
            )
            (self.transition_matrices,  self.observation_matrices,
             self.transition_offsets, self.observation_offsets,
             self.transition_covariance, self.observation_covariance,
             self.initial_state_mean, self.initial_state_covariance) = (
                _em(Z, self.transition_offsets, self.observation_offsets,
                    smoothed_state_means, smoothed_state_covariances,
                    sigma_pair_smooth, given=given
                )
            )
        return self

    def loglikelihood(self, X):
        """Calculate the log likelihood of all observations

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
        (transition_matrices, transition_offsets,
         transition_covariance, observation_matrices,
         observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        # apply the Kalman Filter
        (predicted_state_means, predicted_state_covariance2s,
         filtered_state_means, filtered_state_covariance2s) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance,
                Z
            )
        )

        # get likelihoods for each time step
        predicted_state_covariances = (
            _reconstruct_covariances(predicted_state_covariance2s)
        )
        loglikelihoods = _loglikelihoods(
          observation_matrices, observation_offsets, observation_covariance,
          predicted_state_means, predicted_state_covariances, Z
        )

        return np.sum(loglikelihoods)
