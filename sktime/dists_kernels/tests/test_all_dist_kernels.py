# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sktime.dists_kernels._base import BasePairwiseTransformer
from sktime.utils import all_estimators
from sktime.utils._testing.panel import make_transformer_problem

PAIRWISE_TRANSFORMERS_TAB = all_estimators(
    estimator_types="transformer-pairwise-tabular", return_names=False
)
PAIRWISE_TRANSFORMERS_PAN = all_estimators(
    estimator_types="transformer-pairwise-panel", return_names=False
)

X1 = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=True
)[0]
X2 = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=True
)[0]

X1_df = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=True
)[0]
X2_df = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=True
)[0]


@pytest.mark.parametrize("pairwise_transformers_tab", PAIRWISE_TRANSFORMERS_TAB)
def test_pairwise_transformers_tab(pairwise_transformers_tab):
    _general_pairwise_transformer_tests(X1, X2, pairwise_transformers_tab)
    _general_pairwise_transformer_tests(X1_df, X2_df, pairwise_transformers_tab)


def _general_pairwise_transformer_tests(x, y, pairwise_transformers_tab):
    transformer: BasePairwiseTransformer = pairwise_transformers_tab()

    # test return matrix
    transformation = transformer.transform(X1, X2)

    assert transformer.symmetric is False, (
        f"Symmetric is set to wrong value for {pairwise_transformers_tab} "
        f"when both X1 and X2 passed"
    )
    assert isinstance(
        transformation, np.ndarray
    ), f"Shape of matrix returned is wrong for {pairwise_transformers_tab}"
    assert transformation.shape == (
        4,
        5,
    ), f"Shape of matrix returned is wrong for {pairwise_transformers_tab}"

    # test symmetric
    transformer: BasePairwiseTransformer = pairwise_transformers_tab()
    transformation = transformer.transform(X1)
    for i in range(len(X1)):
        row = (transformation[i, :]).T
        column = transformation[:, i]
        assert np.array_equal(row, column)
    assert transformer.symmetric, (
        f"Symmetric is set to wrong value for {pairwise_transformers_tab} "
        f"when only X1"
    )
