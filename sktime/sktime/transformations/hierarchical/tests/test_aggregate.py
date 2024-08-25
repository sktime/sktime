#!/usr/bin/env python3 -u
"""Tests for hierarchical aggregator."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen


# test for equal output with with named/unnamed indexes
@pytest.mark.skipif(
    not run_test_for_class(Aggregator),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("flatten_single_levels", [True, False])
def test_aggregator_fit_transform_index(flatten_single_levels):
    """Tests fit_transform of aggregator function.

    This test asserts that the output of Aggregator using fit_transform() with a named
    multiindex is equal to an unnamed one. It also tests that Aggregator does not change
    the names of the input index in both cases.
    """
    agg = Aggregator(flatten_single_levels=flatten_single_levels)

    X = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
    )
    # named indexes
    X_agg = agg.fit_transform(X)
    msg = "Aggregator returns wrong index names."
    assert X_agg.index.names == X.index.names, msg

    # unnamed indexes
    X.index.rename([None] * X.index.nlevels, inplace=True)
    X_agg_unnamed = agg.fit_transform(X)
    assert X_agg_unnamed.index.names == X.index.names, msg

    msg = "Aggregator returns different output for named and unnamed indexes."
    assert X_agg.equals(X_agg_unnamed), msg


# test that flatten_single_levels works as expected
@pytest.mark.skipif(
    not run_test_for_class(Aggregator),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_aggregator_flatten():
    """Tests Aggregator flattening single levels.

    This tests that the flatten_single_levels argument works as expected for a fixed
    example of a complicated hierarchy.
    """
    agg = Aggregator(flatten_single_levels=False)
    agg_flat = Aggregator(flatten_single_levels=True)

    X = _bottom_hier_datagen(
        no_bottom_nodes=10,
        no_levels=4,
        random_seed=111,
    )
    # aggregate without flattening
    X_agg = agg.fit_transform(X)
    # aggregate with flattening
    X_agg_flat = agg_flat.fit_transform(X)

    msg = (
        "Aggregator without flattening should have 21 unique levels, "
        "with the time index removed, for random_seed=111."
    )
    assert len(X_agg.droplevel(-1).index.unique()) == 21, msg

    msg = (
        "Aggregator with flattening should have 17 unique levels, "
        "with the time index removed, for random_seed=111."
    )
    assert len(X_agg_flat.droplevel(-1).index.unique()) == 17, msg
