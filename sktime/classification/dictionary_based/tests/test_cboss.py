import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_gunpoint


def test_cboss_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train boss
    cboss = ContractableBOSS(n_parameter_samples=50, max_ensemble_size=10,
                             random_state=0)
    cboss.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cboss.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cboss_gunpoint_probas)


cboss_gunpoint_probas = np.array([
    [0.0, 0.9999999999999998, ],
    [0.2, 0.7999999999999999, ],
    [0.5298890992696782, 0.4701109007303219, ],
    [0.4103327021909657, 0.5896672978090345, ],
    [0.0, 0.9999999999999998, ],
    [0.07011090073032188, 0.9298890992696779, ],
    [0.12988909926967812, 0.8701109007303217, ],
    [0.21033270219096561, 0.7896672978090343, ],
    [0.2, 0.7999999999999999, ],
    [0.12988909926967812, 0.8701109007303217, ],
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
#     cboss = ContractableBOSS(n_parameter_samples=50, max_ensemble_size=10,
#                              random_state=0)
#
#     cboss.fit(X_train.iloc[indices], y_train[indices])
#     probas = cboss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
