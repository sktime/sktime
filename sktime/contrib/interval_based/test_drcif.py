# -*- coding: utf-8 -*-
# import numpy as np
# from numpy import testing
#
# from sktime.contrib.interval_based import DiverseRepresentationCanonicalIntervalForest
# from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions
#
#
# def test_drcif_on_gunpoint():
#     # load gunpoint data
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     # train DrCIF
#     drcif = DiverseRepresentationCanonicalIntervalForest(
#         n_estimators=100,
#         random_state=0,
#     )
#     drcif.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = drcif.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, drcif_gunpoint_probas)
#
#
# def test_drcif_on_power_demand():
#     # load power demand data
#     X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
#     X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(100)
#
#     # train DrCIF
#     drcif = DiverseRepresentationCanonicalIntervalForest(
#         n_estimators=100,
#         random_state=0,
#     )
#     drcif.fit(X_train, y_train)
#
#     score = drcif.score(X_test.iloc[indices], y_test[indices])
#     assert score >= 0.92
#
#
# def test_drcif_on_basic_motions():
#     # load basic motions data
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     # train DrCIF
#     drcif = DiverseRepresentationCanonicalIntervalForest(
#         n_estimators=100,
#         random_state=0,
#     )
#     drcif.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = drcif.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, drcif_basic_motions_probas)
#
#
# drcif_gunpoint_probas = np.array(
#     [
#         [
#             0.15,
#             0.85,
#         ],
#         [
#             0.43,
#             0.57,
#         ],
#         [
#             0.61,
#             0.39,
#         ],
#         [
#             0.42,
#             0.58,
#         ],
#         [
#             0.11,
#             0.89,
#         ],
#         [
#             0.54,
#             0.46,
#         ],
#         [
#             0.31,
#             0.69,
#         ],
#         [
#             0.58,
#             0.42,
#         ],
#         [
#             0.58,
#             0.42,
#         ],
#         [
#             0.18,
#             0.82,
#         ],
#     ]
# )
# drcif_basic_motions_probas = np.array(
#     [
#         [
#             0.98,
#             0.02,
#         ],
#         [
#             0.01,
#             0.99,
#         ],
#         [
#             0.91,
#             0.09,
#         ],
#         [
#             0.04,
#             0.96,
#         ],
#         [
#             0.97,
#             0.03,
#         ],
#         [
#             0.96,
#             0.04,
#         ],
#         [
#             0.02,
#             0.98,
#         ],
#         [
#             0.98,
#             0.02,
#         ],
#         [
#             0.02,
#             0.98,
#         ],
#         [
#             0.04,
#             0.96,
#         ],
#         [
#             0.0,
#             1.0,
#         ],
#         [
#             0.99,
#             0.01,
#         ],
#         [
#             0.01,
#             0.99,
#         ],
#         [
#             0.01,
#             0.99,
#         ],
#         [
#             0.95,
#             0.05,
#         ],
#         [
#             0.96,
#             0.04,
#         ],
#         [
#             0.02,
#             0.98,
#         ],
#         [
#             0.12,
#             0.88,
#         ],
#         [
#             0.98,
#             0.02,
#         ],
#         [
#             1.0,
#             0.0,
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
# #     drcif_u = DiverseRepresentationCanonicalIntervalForest(
# #         n_estimators=100,
# #         random_state=0,
# #     )
# #
# #     drcif_u.fit(X_train.iloc[indices], y_train[indices])
# #     probas = drcif_u.predict_proba(X_test.iloc[indices])
# #     print_array(probas)
# #
# #     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
# #     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
# #     indices = np.random.RandomState(0).permutation(20)
# #
# #     drcif_m = DiverseRepresentationCanonicalIntervalForest(
# #         n_estimators=100,
# #         random_state=0,
# #     )
# #
# #     drcif_m.fit(X_train.iloc[indices], y_train[indices])
# #     probas = drcif_m.predict_proba(X_test.iloc[indices])
# #     print_array(probas)
