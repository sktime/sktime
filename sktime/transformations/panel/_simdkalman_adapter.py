"""Common adapter code for simdkalman.
Shared between Panel and Series transformers"""

__author__ = ["oseiskar"]

import numpy as np


class _SIMDKalmanAdapter:
    def __init__(
        self,
        state_transition,
        process_noise,
        measurement_noise,
        measurement_function,
        initial_state=None,
        initial_state_covariance=None,
        denoising=False,
        hidden=True,
    ):
        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = measurement_function
        self.observation_noise = measurement_noise
        self.initial_state = initial_state
        self.initial_state_covariance = initial_state_covariance

        # check that the parameters are OK
        self._build_kalman_filter()

        self.smooth = denoising
        self.hidden = hidden

    def _build_kalman_filter(self):
        from simdkalman import KalmanFilter as simdkalman_KalmanFilter

        return simdkalman_KalmanFilter(
            state_transition=self.state_transition,
            process_noise=self.process_noise,
            observation_model=self.observation_model,
            observation_noise=self.observation_noise,
        )

    # TODO: support the EM algorithm

    def compute(self, X, multiple_instances):
        if multiple_instances:
            len(X.shape) == 3
        else:
            assert len(X.shape) == 2
            X = X[np.newaxis, ...]

        r = self._build_kalman_filter().compute(
            X,
            n_test=0,
            initial_value=self.initial_state,
            initial_covariance=self.initial_state_covariance,
            observations=not self.hidden,
            states=self.hidden,
            covariances=False,
            filtered=not self.smooth,
            smoothed=self.smooth,
        )

        if self.smooth:
            result = r.smoothed
        else:
            result = r.filtered

        if self.hidden:
            result = result.states
        else:
            result = result.observations

        result = result.mean

        # undo auto-flatten in simdkalman
        if len(result.shape) < 3:
            result = result[..., np.newaxis]

        assert len(result.shape) == 3

        if not multiple_instances:
            assert result.shape[0] == 1
            result = result[0, ...]

        return result
