# -*- coding: utf-8 -*-
"""
HMM Annotation Estimator.

Implements a basic Hidden Markov Model (HMM) as an annotation estimator.
To read more about the algorithm, check out the wikipedia page:
(https://en.wikipedia.org/wiki/Hidden_Markov_model
"""
from typing import Tuple

import numpy as np

from sktime.annotation.base._base import SimpleBaseEstimator

__author__ = ["miraep8"]
__all__ = ["HMM"]


def _calculate_trans_mats(
    initial_probs, emi_probs, transition_prob_mat, num_obs, num_states
) -> Tuple[np.array, np.array]:
    """Calculate the transition mats used in the viterbi algorithm.

    Parameters
    ----------
        - observations: nxm array of n m-dimensional variables.

    Returns
    -------
        - self
    """
    # trans_prob represents the maximum probability of being in that
    # state at that stage
    trans_prob = np.zeros((num_states, num_obs))
    trans_prob[:, 0] = np.log(initial_probs)

    # trans_id is the index of the state that would have been the most
    # likely preceeding state.
    trans_id = np.zeros((num_states, num_obs), dtype=np.int32)

    # use Vertibi Algorithm to fill in trans_prob and trans_id:
    for i in range(1, num_obs):
        # use log probabilities to try to keep nums reasonable -Inf
        # means 0 probability
        paths = np.zeros((num_states, num_states))
        for j in range(num_states):
            paths[j, :] += trans_prob[:, i - 1]  # adds prev trans_prob column-wise
            paths[:, j] += np.log(emi_probs[:, i])  # adds log(probs_sub) row-wise
        paths += np.log(
            transition_prob_mat
        )  # adds log(transition_prob_mat) element-wise
        trans_id[:, i] = np.argmax(paths, axis=0)
        trans_prob[:, i] = np.max(paths, axis=0)

    if np.any(np.isinf(trans_prob[:, -1])):
        raise ValueError("Change parameters, the distribution doesn't work")

    return trans_prob, trans_id


def _make_emission_probs(emission_funcs, observations):
    # assign emission probabilities from each state to each position:

    emi_probs = np.zeros(shape=(len(emission_funcs), len(observations)))
    for state_id, emission_tuple in enumerate(emission_funcs):
        emission_func = emission_tuple[0]
        kwargs = emission_tuple[1]
        emi_probs[state_id, :] = np.array(
            [emission_func(x, **kwargs) for x in observations]
        )
    return emi_probs


class HMM(SimpleBaseEstimator):
    """Implements a simple HMM fitted with viterbi algorithm.

    I will add more details about this estimator here

    Parameters
    ----------
    emission_funcs : a list of functions, should act as PDFs
    transition_prob_mat: a nxn array of probabilities (sum to
        zero across the row)
    initial_probs: optional, a 1d array of probabilities across
        the starting states. Should match prior beliefs.  If none
        is passed will give each state an equal initial probability.
    """

    def __init__(
        self,
        emission_funcs: list,
        transition_prob_mat: np.ndarray,
        initial_probs: np.ndarray = None,
    ):
        if not initial_probs:
            initial_probs = 1.0 / (len(emission_funcs)) * np.ones(len(emission_funcs))
        num_states = len(emission_funcs)
        params = {
            "emission_funcs": emission_funcs,
            "transition_prob_mat": transition_prob_mat,
            "num_states": num_states,
            "states": [i for i in range(num_states)],
            "initial_probs": initial_probs,
            "made_prediction": False,
        }
        self.set_params(**params)

    parameters = {
        "emission_funcs": list,
        "transition_prob_mat": np.ndarray,
        "num_states": int,
        "num_obs": int,
        "states": list,
        "initial_probs": np.ndarray,
        "trans_prob": np.ndarray,
        "trans_id": np.ndarray,
        "made_prediction": bool,
        "hmm_predict": np.ndarray,
    }

    def hmm_viterbi_fit(self) -> np.array:
        """Fit peaks to the provided z_scores data set using the vertibi algorithm.

        Parameters
        ----------
            - observations: nxm array of n m-dimensional variables.

        Returns
        -------
            -hmm: a 2xn array with the first column being position and the second
                column being a peak assignment.
        """
        if self.made_prediction:
            return self.hmm_predict
        hmm_fit = np.zeros(self.num_obs)
        # Now we trace backwards and find the most likely path:
        max_inds = np.zeros(self.num_obs, dtype=np.int32)
        max_inds[-1] = np.argmax(self.trans_prob[:, -1])
        hmm_fit[-1] = self.states[max_inds[-1]]
        for index in reversed(list(range(1, self.num_obs))):
            max_inds[index - 1] = self.trans_id[max_inds[index], index]
            hmm_fit[index - 1] = self.states[max_inds[index - 1]]
        self.made_prediction = True
        self.hmm_predict = hmm_fit
        return hmm_fit

    def verify_params(self):
        """Perform some basic checks that the input behaves as expected."""
        if not len(self.states) == len(self.transition_prob_mat[0]):
            raise ValueError(
                "Number of states should match the",
                " length of the transition matrix.  But sizes provided were",
                f" num states = {self.num_states}, and len transition_prob_mat =",
                f" {len(self.transition_prob_mat)}",
            )

        if not len(self.transition_prob_mat[0]) == len(self.transition_prob_mat):
            raise ValueError("transition_prob_mat must be a square array")

        # add a test that all columns of trans mat sum to 1

    def fit(self, observations: np.ndarray):
        """Calculate the most likely sequence of hidden states."""
        self.verify_params()
        self.num_obs = len(observations)
        emi_probs = _make_emission_probs(self.emission_funcs, observations)
        trans_prob, trans_id = _calculate_trans_mats(
            self.initial_probs,
            emi_probs,
            self.transition_prob_mat,
            self.num_obs,
            self.num_states,
        )
        self.trans_prob = trans_prob
        self.trans_id = trans_id
        return self

    def predict(self):
        """Calculate and return the most likely seq."""
        return self.hmm_viterbi_fit()
