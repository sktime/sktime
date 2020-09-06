import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import WEASEL
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_weasel_on_gunpoint():
    # load gunpoint data

    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    #indices = np.random.RandomState(0).permutation(10)

    # train WEASEL
    weasel = WEASEL(random_state=0)
    weasel.fit(X_train, y_train)

    # assert probabilities are the same
    # probas = weasel.predict_proba(X_test)
    # print_array(probas)
    # testing.assert_array_equal(probas, boss_gunpoint_probas)

    print(weasel.score(X_test, y_test))


def test_weasel_on_power_demand():
    # load gunpoint data

    X_train, y_train = load_italy_power_demand(split='train', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='test', return_X_y=True)

    # train WEASEL
    weasel = WEASEL(random_state=0)
    weasel.fit(X_train, y_train)

    print(weasel.score(X_test, y_test))


boss_gunpoint_probas = np.array([
    [0.0, 1.0, ],
    [0.15384615384615385, 0.8461538461538461, ],
    [1.0, 0.0, ],
    [0.9230769230769231, 0.07692307692307693, ],
    [0.0, 1.0, ],
    [0.8461538461538461, 0.15384615384615385, ],
    [0.0, 1.0, ],
    [0.8461538461538461, 0.15384615384615385, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
])


def print_array(array):
    print('[')
    for sub_array in array:
        print('[', end='')
        for value in sub_array:
            print(value.astype(str), end='')
            print(', ', end='')
        print('],')
    print(']')

# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split='train', return_X_y=True)
#     X_test, y_test = load_gunpoint(split='test', return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     boss = BOSSEnsemble(random_state=0)
#     indiv_boss = BOSSIndividual(random_state=0)
#
#     boss.fit(X_train.iloc[indices], y_train[indices])
#     probas = boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_boss.fit(X_train.iloc[indices], y_train[indices])
#     probas = indiv_boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
