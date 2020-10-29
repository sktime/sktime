# -*- coding: utf-8 -*-
# import numpy as np
# from numpy import testing
#
# from sktime.contrib.hybrid._catch22_forest_classifier import Catch22ForestClassifier
# from sktime.datasets import load_gunpoint, load_basic_motions
#
#
# def test_catch22_forest_classifier_on_gunpoint():
#     # load gunpoint data
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     # train c22f
#     c22f = Catch22ForestClassifier(random_state=0)
#     c22f.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = c22f.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, catch22_forest_classifier_gunpoint_probas)
#
#
# def test_catch22_forest_classifier_on_basic_motions():
#     # load basic motions data
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     # train c22f
#     c22f = Catch22ForestClassifier(random_state=0)
#     c22f.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = c22f.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, catch22_forest_classifier_basic_motions_probas)
#
#
# catch22_forest_classifier_gunpoint_probas = np.array(
#     [
#         [
#             0.07,
#             0.93,
#         ],
#         [
#             0.35,
#             0.65,
#         ],
#         [
#             0.35,
#             0.65,
#         ],
#         [
#             0.13,
#             0.87,
#         ],
#         [
#             0.11,
#             0.89,
#         ],
#         [
#             0.79,
#             0.21,
#         ],
#         [
#             0.05,
#             0.95,
#         ],
#         [
#             0.59,
#             0.41,
#         ],
#         [
#             0.56,
#             0.44,
#         ],
#         [
#             0.06,
#             0.94,
#         ],
#     ]
# )
# catch22_forest_classifier_basic_motions_probas = np.array(
#     [
#         [
#             0.99,
#             0.01,
#         ],
#         [
#             0.16,
#             0.84,
#         ],
#         [
#             0.35,
#             0.65,
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
#             0.97,
#             0.03,
#         ],
#         [
#             0.15,
#             0.85,
#         ],
#         [
#             0.94,
#             0.06,
#         ],
#         [
#             0.15,
#             0.85,
#         ],
#         [
#             0.01,
#             0.99,
#         ],
#         [
#             0.04,
#             0.96,
#         ],
#         [
#             0.86,
#             0.14,
#         ],
#         [
#             0.01,
#             0.99,
#         ],
#         [
#             0.03,
#             0.97,
#         ],
#         [
#             0.43,
#             0.57,
#         ],
#         [
#             0.96,
#             0.04,
#         ],
#         [
#             0.22,
#             0.78,
#         ],
#         [
#             0.36,
#             0.64,
#         ],
#         [
#             0.35,
#             0.65,
#         ],
#         [
#             0.94,
#             0.06,
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
# # if __name__ == "__main__":
# #     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
# #     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
# #     indices = np.random.RandomState(0).permutation(10)
# #
# #     c22f_u = Catch22ForestClassifier(random_state=0)
# #
# #     c22f_u.fit(X_train.iloc[indices], y_train[indices])
# #     probas = c22f_u.predict_proba(X_test.iloc[indices])
# #     print_array(probas)
# #
# #     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
# #     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
# #     indices = np.random.RandomState(0).permutation(20)
# #
# #     c22f_m = Catch22ForestClassifier(random_state=0)
# #
# #     c22f_m.fit(X_train.iloc[indices], y_train[indices])
# #     probas = c22f_m.predict_proba(X_test.iloc[indices])
# #     print_array(probas)
