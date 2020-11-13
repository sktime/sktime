# -*- coding: utf-8 -*-
# import numpy as np
# from numpy import testing
#
# from sktime.contrib.interval_based._cif import CanonicalIntervalForest
# from sktime.datasets import load_gunpoint, load_italy_power_demand
#
#
# def test_cif_on_gunpoint():
#     # load gunpoint data
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     # train CIF
#     cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
#     cif.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = cif.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, cif_gunpoint_probas)
#
#
# def test_cif_on_power_demand():
#     # load power demand data
#     X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
#     X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(100)
#
#     # train CIF
#     cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
#     cif.fit(X_train, y_train)
#
#     score = cif.score(X_test.iloc[indices], y_test[indices])
#     assert score >= 0.92
#
#
# cif_gunpoint_probas = np.array(
#     [
#         [
#             0.12,
#             0.88,
#         ],
#         [
#             0.5,
#             0.5,
#         ],
#         [
#             0.57,
#             0.43,
#         ],
#         [
#             0.41,
#             0.59,
#         ],
#         [
#             0.06,
#             0.94,
#         ],
#         [
#             0.59,
#             0.41,
#         ],
#         [
#             0.25,
#             0.75,
#         ],
#         [
#             0.62,
#             0.38,
#         ],
#         [
#             0.57,
#             0.43,
#         ],
#         [
#             0.11,
#             0.89,
#         ],
#     ]
# )
#
#
# # def print_array(array):
# #     print('[')
# #     for sub_array in array:
# #         print('[')
# #         for value in sub_array:
# #             print(value.astype(str), end='')
# #             print(', ')
# #         print('],')
# #     print(']')
# #
# #
# # if __name__ == "__main__":
# #     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
# #     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
# #     indices = np.random.RandomState(0).permutation(10)
# #
# #     cif_u = CanonicalIntervalForest(n_estimators=100, random_state=0)
# #
# #     cif_u.fit(X_train.iloc[indices], y_train[indices])
# #     probas = cif_u.predict_proba(X_test.iloc[indices])
# #     print_array(probas)
