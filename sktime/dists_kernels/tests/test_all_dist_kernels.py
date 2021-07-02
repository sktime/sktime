# -*- coding: utf-8 -*-
import pytest

from sktime.dists_kernels._base import BasePairwiseTransformer
from sktime.utils import all_estimators
from sktime.utils._testing.panel import make_transformer_problem

PAIRWISE_TRANSFORMERS_TAB = all_estimators(
    estimator_types="transformer-pairwise-tab", return_names=False
)
PAIRWISE_TRANSFORMERS_PAN = all_estimators(
    estimator_types="transformer-pairwise-ts", return_names=False
)

# testing data
X1 = make_transformer_problem(
    n_instances=1, n_columns=2, random_state=1, return_numpy=True
)[0]
X2 = make_transformer_problem(
    n_instances=1, n_columns=2, random_state=2, return_numpy=True
)[0]
X1_df = make_transformer_problem(
    n_instances=1, n_columns=1, random_state=1, return_numpy=False
)
X2_df = make_transformer_problem(
    n_instances=1, n_columns=1, random_state=2, return_numpy=False
)


@pytest.mark.parametrize("pairwise_transformers_tab", PAIRWISE_TRANSFORMERS_TAB)
def test_pairwise_transformers_tab(pairwise_transformers_tab):
    transformer: BasePairwiseTransformer = pairwise_transformers_tab()
    tranformation = transformer.transform(X1, X2)
    return tranformation


# @pytest.mark.parametrize("pairwise_transformers_pan", PAIRWISE_TRANSFORMERS_PAN)
# def test_pairwise_transformers_tab(pairwise_transformers_pan):
#     transformer = pairwise_transformers_pan(X1, X2)
#     test = transformer
#     print(test)
#     print(type(test))


"""
Behaviour: returns pairwise distance/kernel matrix
    between samples in X and X2
        if X2 is not passed, is equal to X

alias for transform

Parameters
----------
X: pd.DataFrame of length n, or 2D np.array with n rows
X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
    default X2 = X

Returns
-------
distmat: np.array of shape [n, m]
    (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]

Writes to self
--------------
symmetric: bool = True if X2 was not passed, False if X2 was passed
    for use to make internal calculations efficient, e.g., in _transform
"""
