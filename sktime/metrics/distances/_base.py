# -*- coding: utf-8 -*-
"""
Abstract base class for pairwise transformers (such as distance/kernel matrix makers)
"""
#
# __author__ = ["fkiraly"]
#
# import numpy as np
# import pandas as pd
#
# from sktime.base import BaseEstimator
#
# from sktime.utils.validation.series import check_series
#
#
# class BasePairwiseTransformer:
#
#     def __init__(self):
#         super().__init__()
#         self.X_equals_X2 = False
#
#     def transform(self, X, X2=None):
#         X = check_series(X)
#
#         if X2 is None:
#             X2 = X
#             self.X_equals_X2 = True
#         else:
#             X2 = check_series(X2)
#
#             def input_as_numpy(val):
#                 if isinstance(val, pd.DataFrame):
#                     return val.to_numpy(copy=True)
#                 return val
#
#             temp_X = input_as_numpy(X)
#             temp_X2 = input_as_numpy(X2)
#             if np.array_equal(temp_X, temp_X2):
#                 self.X_equals_X2 = True
#
#         return self._transform(X=X, X2=X2)
#
#     def _transform(self, X, X2=None):
#         raise NotImplementedError
#
#
# class BasePairwiseTransformerPanel():
#
#     def __init__(self):
#         super(BasePairwiseTransformerPanel, self).__init__()
#         self.X_equals_X2 = False
#
#     def transform(self, X, X2=None):
#         X = _pairwise_panel_x_check(X)
#
#         if X2 is None:
#             X2 = X
#             self.X_equals_X2 = True
#         else:
#             X2 = _pairwise_panel_x_check(X2)
#             if len(X2) == len(X):
#                 for i in range(len(X2)):
#                     if not X[i].equals(X2[i]):
#                         break
#                 self.X_equals_X2 = True
#
#         return self._transform(X=X, X2=X2)
#
#     def _transform(self, X, X2=None):
#         raise NotImplementedError
