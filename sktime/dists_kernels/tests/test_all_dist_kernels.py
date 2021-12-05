# -*- coding: utf-8 -*-
"""Tests for pairwise transformer."""

import numpy as np
import pytest

from sktime.datatypes import convert_to
from sktime.registry import all_estimators
from sktime.utils._testing.panel import make_transformer_problem
from sktime.utils._testing.series import _make_series

PAIRWISE_TRANSFORMERS = all_estimators(
    estimator_types="transformer-pairwise", return_names=False
)
PAIRWISE_TRANSFORMERS_PANEL = all_estimators(
    estimator_types="transformer-pairwise-panel", return_names=False
)

EXPECTED_SHAPE = (4, 5)

X1_tab = _make_series(
    n_columns=4,
    n_timepoints=4,
    random_state=1,
    return_numpy=True,
)
X2_tab = _make_series(
    n_columns=4,
    n_timepoints=5,
    random_state=2,
    return_numpy=True,
)

X1_tab_df = _make_series(
    n_columns=4,
    n_timepoints=4,
    random_state=1,
    return_numpy=False,
)
X2_tab_df = _make_series(
    n_columns=4,
    n_timepoints=5,
    random_state=2,
    return_numpy=False,
)

VALID_INPUTS_TABULAR = [(X1_tab, X2_tab), (X1_tab_df, X2_tab_df)]


@pytest.mark.parametrize("x,y", VALID_INPUTS_TABULAR)
@pytest.mark.parametrize("pairwise_transformer", PAIRWISE_TRANSFORMERS)
def test_pairwise_transformers_tabular(x, y, pairwise_transformer):
    """Main test function for pairwise transformers on tabular data."""
    _general_pairwise_transformer_tests(x, y, pairwise_transformer)


X1_list_df = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=False
)
X2_list_df = make_transformer_problem(
    n_instances=5, n_columns=4, n_timepoints=5, random_state=2, return_numpy=False
)

X1_num_pan = convert_to(X1_list_df, to_type="numpy3D")
X2_num_pan = convert_to(X2_list_df, to_type="numpy3D")


VALID_INPUTS_PANEL = [
    (X1_num_pan, X2_num_pan),
    (X1_list_df, X2_list_df),
]


@pytest.mark.parametrize("x,y", VALID_INPUTS_PANEL)
@pytest.mark.parametrize("pairwise_transformer", PAIRWISE_TRANSFORMERS_PANEL)
def test_pairwise_transformers_panel(x, y, pairwise_transformer):
    """Main test function for pairwise transformers on panel data."""
    _general_pairwise_transformer_tests(x, y, pairwise_transformer)


def _general_pairwise_transformer_tests(x, y, pairwise_transformer):
    # test return matrix
    transformer = pairwise_transformer.create_test_instance()
    transformation = transformer.transform(x, y)

    assert transformer.X_equals_X2 is False, (
        f"X_equals_X2 is set to wrong value for {pairwise_transformer} "
        f"when both X1 and X2 passed"
    )
    assert isinstance(
        transformation, np.ndarray
    ), f"Shape of matrix returned is wrong for {pairwise_transformer}"
    assert (
        transformation.shape == EXPECTED_SHAPE
    ), f"Shape of matrix returned is wrong for {pairwise_transformer}"
    assert transformer.X_equals_X2 is False, (
        f"X_equals_X2 is set to wrong value for {transformer} " f"when only X passed"
    )

    transformer = pairwise_transformer.create_test_instance()
    transformation = transformer.transform(x)
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
