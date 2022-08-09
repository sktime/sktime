# -*- coding: utf-8 -*-
"""MUSE test code."""
from sktime.classification.dictionary_based._muse import MUSE
from sktime.datasets import load_basic_motions


def test_muse_train_estimate():
    """Test of TDE train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    # train MUSE
    muse = MUSE(random_state=0, n_jobs=4)

    # import time

    # start = time.time()
    muse.fit(X_train, y_train)
    # print("Fit", time.time() - start)
    # start = time.time()
    # print(muse.score(X_test, y_test))
    # print("Score", time.time() - start)
