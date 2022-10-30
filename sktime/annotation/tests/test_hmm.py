# -*- coding: utf-8 -*-
"""Tests for HMM annotation estimator."""

__author__ = ["miraep8"]

import numpy as np
import pytest
from numpy import array_equal, asarray
from scipy.stats import norm

from sktime.annotation.hmm import HMM


def test_hmm_basic_gauss():
    """Test HMM works for basic Gaussian synthetic data.

    Check if the predicted change points match.
    """
    centers = [3.5, -5, 8, 10, 0.5]
    seg_lengths = [100, 50, 25, 70, 15]
    gauss_data = np.zeros([sum(seg_lengths)])
    labels = np.zeros([sum(seg_lengths)])
    i = 0
    for ind, s_len in enumerate(seg_lengths):
        gauss_data[i : i + s_len] = np.random.default_rng().normal(
            loc=centers[ind], scale=0.25, size=[s_len]
        )
        labels[i : i + s_len] = np.asarray([ind for _ in range(s_len)])
        i += s_len
    emi_funcs = [(norm.pdf, {"loc": mean, "scale": 1}) for mean in centers]
    transition_matrix = np.asarray(
        [
            [((1 / (s_len / 14)) / (len(centers) - 1)) for state in centers]
            for s_len in seg_lengths
        ]
    )
    for ind, s_len in enumerate(seg_lengths):
        transition_matrix[ind, ind] = 1 - (1 / (s_len / 14))
    gauss_test = HMM(emi_funcs, transition_matrix)
    gauss_test.fit(gauss_data)
    predicted_labels = gauss_test.predict(gauss_data)
    assert len(predicted_labels == labels) >= 0.95 * len(predicted_labels)


def test_reject_bad_inputs():
    """Demonstrate failute when initialized with bad inputs."""
    # test we get a ValueError if we don't use square trans_prob
    size = 2
    valid_emi_funcs = [(norm.pdf, {"loc": 0, "scale": 1}) for _ in range(size)]
    # trans_mat must be square:
    with pytest.raises(ValueError):
        HMM(
            emission_funcs=valid_emi_funcs,
            transition_prob_mat=asarray([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]),
        )
    # trans_mat rows must sum to 1:
    with pytest.raises(ValueError):
        HMM(
            emission_funcs=valid_emi_funcs,
            transition_prob_mat=asarray([[10, 10], [0.2, 0.8]]),
        )
    # emi_funcs and trans mat must have shared dimension:
    with pytest.raises(ValueError):
        HMM(
            emission_funcs=valid_emi_funcs,
            transition_prob_mat=asarray(
                [[0.25, 0.25, 0.5], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]]
            ),
        )


def test_hmm_behaves_as_expected_on_simple_input():
    """Test HMM is consistent with expected ground truth in simple ex."""
    # define the emission probs for our HMM model:
    centers = [3.5, -5]
    sd = [0.25 for i in centers]
    emi_funcs = [
        (norm.pdf, {"loc": mean, "scale": sd[ind]}) for ind, mean in enumerate(centers)
    ]
    hmm_est = HMM(emi_funcs, asarray([[0.25, 0.75], [0.666, 0.333]]))
    # generate synthetic data (or of course use your own!)
    obs = asarray([3.7, 3.2, 3.4, 3.6, -5.1, -5.2, -4.9])
    hmm_est = hmm_est.fit(obs)
    labels = hmm_est.predict(obs)
    ground_truth = asarray([0, 0, 0, 0, 1, 1, 1])
    assert array_equal(labels, ground_truth)
