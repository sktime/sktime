# -*- coding: utf-8 -*-
"""Test the MUSE multivariate classifier."""

import numpy as np
from sktime.classification.dictionary_based import MUSE
from sktime.datasets import load_japanese_vowels


def test_muse_on_japanese_vowels():
    """Test MUSE classifier based on accuracy on JapaneseVowels."""
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(50)

    # train WEASEL+MUSE on multivariate data
    muse = MUSE(random_state=1379, window_inc=4, use_first_order_differences=False)
    muse.fit(X_train.iloc[indices], y_train[indices])

    score = muse.score(X_test.iloc[indices], y_test[indices])
    # print(score)
    assert score >= 0.99
