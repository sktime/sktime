# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sktime.dists_kernels._base import BasePairwiseTransformer
from sktime.utils import all_estimators
from sktime.utils._testing.panel import make_transformer_problem
from sktime.tests._config import ESTIMATOR_TEST_PARAMS

PAIRWISE_TRANSFORMERS_TAB = all_estimators(
    estimator_types="transformer-pairwise-tabular", return_names=False
)
PAIRWISE_TRANSFORMERS_PAN = all_estimators(
    estimator_types="transformer-pairwise-panel", return_names=False
)

X1_tab = make_transformer_problem(
    n_instances=4,
    n_columns=4,
    n_timepoints=5,
    random_state=1,
    return_numpy=True,
    panel=False,
)
X2_tab = make_transformer_problem(
    n_instances=5,
    n_columns=5,
    n_timepoints=5,
    random_state=2,
    return_numpy=True,
    panel=False,
)

X1_tab_df = make_transformer_problem(
    n_instances=4,
    n_columns=4,
    n_timepoints=5,
    random_state=1,
    return_numpy=False,
    panel=False,
)
X2_tab_df = make_transformer_problem(
    n_instances=5,
    n_columns=5,
    n_timepoints=5,
    random_state=2,
    return_numpy=True,
    panel=False,
)


@pytest.mark.parametrize("pairwise_transformers_tab", PAIRWISE_TRANSFORMERS_TAB)
def test_pairwise_transformers_tab(pairwise_transformers_tab):
    # Test numpy tabular
    _general_pairwise_transformer_tests(X1_tab, X2_tab, pairwise_transformers_tab)

    # Test dataframe tabular
    _general_pairwise_transformer_tests(X1_tab_df, X2_tab_df, pairwise_transformers_tab)


X1_num_pan = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=True
)
X2_num_pan = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=True
)

X1_list_df = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=False
)
X2_list_df = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=False
)

X1_num_df = np.array(X1_list_df)
X2_num_df = np.array(X2_list_df)


@pytest.mark.parametrize("pairwise_transformers_pan", PAIRWISE_TRANSFORMERS_PAN)
def test_pairwise_transformers_pan(pairwise_transformers_pan):
    test_params = ESTIMATOR_TEST_PARAMS[pairwise_transformers_pan]

    # Test numpy panel (3d numpy)
    _general_pairwise_transformer_tests(
        X1_num_pan, X2_num_pan, pairwise_transformers_pan, test_params
    )

    # Test list of dataframes
    _general_pairwise_transformer_tests(
        X1_list_df, X2_list_df, pairwise_transformers_pan, test_params
    )

    # Test numpy of dataframes
    _general_pairwise_transformer_tests(
        X1_num_df, X2_num_df, pairwise_transformers_pan, test_params
    )


def _general_pairwise_transformer_tests(x, y, pairwise_transformers_tab, kwargs=None):
    if kwargs is not None:
        transformer = pairwise_transformers_tab(**kwargs)
    else:
        transformer: BasePairwiseTransformer = pairwise_transformers_tab()

    # test return matrix
    transformation = transformer.transform(x, y)

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
    if kwargs is not None:
        transformer = pairwise_transformers_tab(**kwargs)
    else:
        transformer: BasePairwiseTransformer = pairwise_transformers_tab()

    transformation = transformer.transform(x)
    for i in range(len(x)):
        row = (transformation[i, :]).T
        column = transformation[:, i]
        assert np.array_equal(row, column)
    assert transformer.symmetric, (
        f"Symmetric is set to wrong value for {pairwise_transformers_tab} "
        f"when only X1"
    )
