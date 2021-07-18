# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sktime.utils import all_estimators
from sktime.utils._testing.panel import make_transformer_problem
from sktime.tests._config import ESTIMATOR_TEST_PARAMS

PAIRWISE_TRANSFORMERS = all_estimators(
    estimator_types="transformer-pairwise", return_names=False
)
PAIRWISE_TRANSFORMERS_PANEL = all_estimators(
    estimator_types="transformer-pairwise-panel", return_names=False
)

EXPECTED_SHAPE = (4, 5)

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


@pytest.mark.parametrize("pairwise_transformers", PAIRWISE_TRANSFORMERS)
def test_pairwise_transformers_tabular(pairwise_transformers):
    # Test numpy tabular
    _general_pairwise_transformer_tests(X1_tab, X2_tab, pairwise_transformers)

    # Test dataframe tabular
    _general_pairwise_transformer_tests(X1_tab_df, X2_tab_df, pairwise_transformers)


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


@pytest.mark.parametrize("pairwise_transformers", PAIRWISE_TRANSFORMERS_PANEL)
def test_pairwise_transformers_panel(pairwise_transformers):
    test_params = ESTIMATOR_TEST_PARAMS[pairwise_transformers]

    # Test numpy panel (3d numpy)
    _general_pairwise_transformer_tests(
        X1_num_pan, X2_num_pan, pairwise_transformers, test_params
    )

    # Test list of dataframes
    _general_pairwise_transformer_tests(
        X1_list_df, X2_list_df, pairwise_transformers, test_params
    )

    # Test numpy of dataframes
    _general_pairwise_transformer_tests(
        X1_num_df, X2_num_df, pairwise_transformers, test_params
    )


def _general_pairwise_transformer_tests(x, y, pairwise_transformers, kwargs=None):
    def create_transformer():
        if kwargs is not None:
            return pairwise_transformers(**kwargs)
        else:
            return pairwise_transformers()

    # test return matrix
    transformer = create_transformer()
    transformation = transformer.transform(x, y)

    assert transformer.X_equals_X2 is False, (
        f"X_equals_X2 is set to wrong value for {pairwise_transformers} "
        f"when both X1 and X2 passed"
    )
    assert isinstance(
        transformation, np.ndarray
    ), f"Shape of matrix returned is wrong for {pairwise_transformers}"
    assert (
        transformation.shape == EXPECTED_SHAPE
    ), f"Shape of matrix returned is wrong for {pairwise_transformers}"
    assert transformer.X_equals_X2 is False, (
        f"X_equals_X2 is set to wrong value for {transformer} " f"when only X passed"
    )

    transformer = create_transformer()
    transformation = transformer.transform(x)
    _x_equals_x2_test(transformation, x, transformer)

    transformer = create_transformer()
    transformation = transformer.transform(x, x)
    _x_equals_x2_test(transformation, x, transformer)


def _x_equals_x2_test(transformation, x, transformer):
    # test X_equals_X2
    for i in range(len(x)):
        # Have to round or test breaks on github (even though works locally)
        row = np.around((transformation[i, :]).T, decimals=5).astype(np.float)
        column = np.around(transformation[:, i], decimals=5).astype(np.float)
        assert np.array_equal(row, column)
    assert transformer.X_equals_X2, (
        f"X_equals_X2 is set to wrong value for {transformer} " f"when only X passed"
    )
