# -*- coding: utf-8 -*-
"""Test the MUSE multivariate classifier."""

import numpy as np
from sktime.classification.dictionary_based import MUSE
from sktime.datasets import load_basic_motions


def test_muse_on_basic_motions():
    """Test MUSE classifier based on accuracy on BasicMotions."""
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train WEASEL+MUSE on multivariate data
    muse = MUSE(random_state=1379, window_inc=4, use_first_order_differences=False)
    muse.fit(X_train.iloc[indices], y_train[indices])

    score = muse.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.99
