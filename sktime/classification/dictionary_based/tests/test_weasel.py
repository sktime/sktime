# -*- coding: utf-8 -*-
"""WEASEL test code."""
from sktime.classification.dictionary_based._weasel import WEASEL
from sktime.datasets import load_gunpoint


def test_weasel_train_estimate():
    """Test of TDE train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    # train weasel
    weasel = WEASEL(random_state=0, n_jobs=-1)

    # import time

    # start = time.time()
    weasel.fit(X_train, y_train)
    # print("Fit", time.time() - start)
    # start = time.time()
    # print(weasel.score(X_test, y_test))
    # print("Score", time.time() - start)
