import numpy as np
from numpy import testing

from sktime.classification.dictionary_based._tde import \
    TemporalDictionaryEnsemble, IndividualTDE
from sktime.datasets import load_gunpoint


def test_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train tde
    tde = TemporalDictionaryEnsemble(random_state=0)
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tde_gunpoint_probas)


def test_individual_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train individual tde
    indiv_tde = IndividualTDE(random_state=0)
    indiv_tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = indiv_tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, individual_tde_gunpoint_probas)


tde_gunpoint_probas = np.array([
    [0.0, 1.0, ],
    [0.05263157894736842, 0.9473684210526315, ],
    [0.8421052631578947, 0.15789473684210525, ],
    [0.8947368421052632, 0.10526315789473684, ],
    [0.0, 1.0, ],
    [0.7368421052631579, 0.2631578947368421, ],
    [0.0, 1.0, ],
    [0.8947368421052632, 0.10526315789473684, ],
    [0.7368421052631579, 0.2631578947368421, ],
    [0.0, 1.0, ],
])
individual_tde_gunpoint_probas = np.array([
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
])


# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[', end='')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(', ', end='')
#         print('],')
#     print(']')
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split='train', return_X_y=True)
#     X_test, y_test = load_gunpoint(split='test', return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     tde = TemporalDictionaryEnsemble(random_state=0)
#     indiv_tde = IndividualTDE(random_state=0)
#
#     tde.fit(X_train.iloc[indices], y_train[indices])
#     probas = tde.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_tde.fit(X_train.iloc[indices], y_train[indices])
#     probas = indiv_tde.predict_proba(X_test.iloc[indices])
#     print_array(probas)
