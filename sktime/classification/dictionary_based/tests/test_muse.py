# import numpy as np
# from numpy import testing

from sktime.classification.dictionary_based import MUSE
from sktime.datasets.base import load_japanese_vowels


def test_muse_on_japanese_vowels():
    # load japanese vowels data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split='test', return_X_y=True)
    # indices = np.random.RandomState(0).permutation(10)

    # train WEASEL+MUSE
    muse = MUSE(random_state=1379, window_inc=8)
    muse.fit(X_train, y_train)

    score = muse.score(X_test, y_test)
    print(score)
    # assert (score >= 0.99)
