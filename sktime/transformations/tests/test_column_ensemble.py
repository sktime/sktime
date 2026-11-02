# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for column-wise application of a single transformer."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.boxcox import BoxCoxTransformer
from sktime.transformations.compose import ColumnEnsembleTransformer
from sktime.transformations.exponent import ExponentTransformer


@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("transformer_class", [BoxCoxTransformer, ExponentTransformer])
def test_single_transformer_matches_explicit_columns(transformer_class, n_columns):
    """A single blueprint is independently fitted and applied to every column."""
    X = pd.DataFrame(
        {
            "b": [1.0, 2.0, 4.0, 8.0],
            "a": [2.0, 5.0, 6.0, 9.0],
            "c": [1.0, 3.0, 5.0, 7.0],
        },
        index=pd.Index([2, 4, 6, 8], name="time"),
    ).iloc[:, :n_columns]
    X_before = X.copy(deep=True)
    blueprint = transformer_class()
    params_before = blueprint.get_params(deep=True)
    transformer = ColumnEnsembleTransformer(blueprint)
    explicit = ColumnEnsembleTransformer(
        [(str(col), blueprint, [col]) for col in X.columns]
    )

    actual = transformer.fit_transform(X)
    expected = explicit.fit_transform(X)

    assert_frame_equal(actual, expected)
    assert_frame_equal(X, X_before)
    assert actual.columns.equals(X.columns)
    assert actual.index.equals(X.index)
    assert not blueprint.is_fitted
    assert blueprint.get_params(deep=True) == params_before
    fitted = [est for _, est, _ in transformer.transformers_]
    assert len(fitted) == n_columns
    assert len({id(est) for est in fitted}) == n_columns
    assert all(est is not blueprint and est.is_fitted for est in fitted)


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("with_remainder", [False, True])
def test_single_transformer_get_params(deep, with_remainder):
    """Single-transformer parameter discovery includes standard nested parameters."""
    blueprint = ExponentTransformer(power=2)
    remainder = ExponentTransformer(power=3) if with_remainder else None
    transformer = ColumnEnsembleTransformer(blueprint, remainder=remainder)

    params = transformer.get_params(deep=deep)

    assert params["transformers"] is blueprint
    assert params["remainder"] is remainder
    assert ("transformers__power" in params) == deep
    assert ("remainder__power" in params) == (deep and with_remainder)
    if deep:
        assert params["transformers__power"] == 2
        if with_remainder:
            assert params["remainder__power"] == 3


@pytest.mark.parametrize(
    "change", ["nested", "replace", "direct_to_list", "list_to_direct", "named"]
)
def test_column_ensemble_parameter_changes(change):
    """Parameter changes preserve direct or explicit column-selection semantics."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    blueprint = ExponentTransformer(power=1)
    explicit = [("selected", blueprint, ["a"])]
    initial = explicit if change in ["list_to_direct", "named"] else blueprint
    transformer = ColumnEnsembleTransformer(initial)

    if change == "nested":
        transformer.set_params(transformers__power=2)
    elif change == "replace":
        transformer.set_params(transformers=ExponentTransformer(power=2))
    elif change == "direct_to_list":
        transformer.set_params(
            transformers=[("selected", ExponentTransformer(power=2), ["a"])]
        )
    elif change == "list_to_direct":
        transformer.set_params(transformers=ExponentTransformer(power=2))
    else:
        transformer.set_params(selected=ExponentTransformer(power=2))

    selected_only = change in ["direct_to_list", "named"]
    expected = X[["a"]] ** 2 if selected_only else X**2
    assert_frame_equal(transformer.fit_transform(X), expected)
    assert isinstance(transformer.transformers, list) == selected_only
    if selected_only:
        assert transformer.transformers[0][0] == "selected"
        assert transformer.transformers[0][2] == ["a"]


def test_single_transformer_remainder_nested_params():
    """Setting nested remainder parameters does not lose the single blueprint."""
    transformer = ColumnEnsembleTransformer(
        ExponentTransformer(power=2), remainder=ExponentTransformer(power=3)
    )

    transformer.set_params(remainder__power=4)

    assert transformer.get_params()["remainder__power"] == 4
    assert transformer.get_params()["transformers__power"] == 2
