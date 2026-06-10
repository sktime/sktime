"""Test Truncator transformer."""

import pytest

from sktime.datasets import load_basic_motions
from sktime.datatypes import get_examples
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.truncation import TruncationTransformer
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.mark.skipif(
    not run_test_for_class(TruncationTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lower, upper, error, message",
    [
        (-1, None, ValueError, TruncationTransformer.error_messages["lower_gt_0"]),
        (-1, 2, ValueError, TruncationTransformer.error_messages["lower_gt_0"]),
        (7, 2, ValueError, TruncationTransformer.error_messages["upper_gt_lower"]),
    ],
)
def test_truncation_constructor_arg_validation(lower, upper, error, message):
    """Test validation of truncation transformer constructor arguments."""
    with pytest.raises(error, match=message):
        tt = TruncationTransformer(lower=lower, upper=upper)  # noqa: F841


@pytest.mark.skipif(
    not run_test_for_class(TruncationTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lower, upper, expected_length",
    [
        (None, None, 60),  # truncate to shortest length
        (5, None, 20),  # truncate to [lower, min_length]
        (None, 7, 56),  # truncate to [0, upper)
        (2, 7, 40),  # truncate to range
    ],
)
def test_truncation_transformer_multi_index(lower, upper, expected_length):
    """Test truncation to the shortest series length in hierarchical data."""
    X = _make_hierarchical(
        same_cutoff=False, max_timepoints=12, min_timepoints=6, random_state=123
    )
    tt = TruncationTransformer(lower=lower, upper=upper)
    Xt = tt.fit_transform(X)

    assert X.shape == (70, 1)
    assert Xt.shape[0] == expected_length


@pytest.mark.skipif(
    not run_test_for_class(TruncationTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_truncation_transformer():
    """Test truncation to the shortest series length."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    truncated_transformer = TruncationTransformer(upper=5)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've truncated them all to 5 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 5 * 6


@pytest.mark.skipif(
    not run_test_for_class(TruncationTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_truncation_parameterised_transformer():
    """Test truncation to the a user defined length."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    truncated_transformer = TruncationTransformer(2, 10)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 8 * 6


@pytest.mark.skipif(
    not run_test_for_class(TruncationTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_truncation_pd_multiindex():
    """Test that column and index names in a pd-multiindex container are preserved."""
    # get a multiindex dataframe, ensure instance levels are string, not int
    X = get_examples("pd-multiindex")[0].copy()
    X.index = X.index.set_levels(["a", "b", "c"], level=0)

    t = TruncationTransformer(1, 2)
    Xt = t.fit_transform(X)

    # assert that column names and index names are preserved
    assert (X.index.get_level_values(0).unique().values == ["a", "b", "c"]).all()
    assert (Xt.columns == X.columns).all()
