# -*- coding: utf-8 -*-
"""Tests for HMM annotation estimator."""

__author__ = ["miraep8"]

import numpy as np
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
    predicted_labels = gauss_test.predict()
    assert len(predicted_labels == labels) >= 0.9 * len(predicted_labels)
