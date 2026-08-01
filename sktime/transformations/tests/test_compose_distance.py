"""Tests for DistanceFeatures."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["shivamlalakiya"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose_distance import DistanceFeatures
from sktime.utils._testing.hierarchical import _make_hierarchical


def _expected_distmat(X):
    """Euclidean distance between all pairs of instances, computed directly."""
    insts = X.index.droplevel(-1).unique()
    distmat = np.zeros((len(insts), len(insts)))
    for i, inst_i in enumerate(insts):
        for j, inst_j in enumerate(insts):
            flat_i = X.loc[inst_i].to_numpy().flatten(order="F")
            flat_j = X.loc[inst_j].to_numpy().flatten(order="F")
            distmat[i, j] = np.linalg.norm(flat_i - flat_j)
    return distmat


@pytest.mark.skipif(
    not run_test_for_class(DistanceFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("hierarchy_levels", [(3,), (2, 2), (2, 2, 2)])
def test_distance_features_hierarchical(hierarchy_levels):
    """Test that hierarchical input is handled, and distances are correct, see #8077."""
    X = _make_hierarchical(
        hierarchy_levels=hierarchy_levels,
        n_columns=2,
        min_timepoints=5,
        max_timepoints=5,
        random_state=42,
    )

    Xt = DistanceFeatures().fit_transform(X)

    inst_ind = X.index.droplevel(-1).unique()

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (len(inst_ind), len(inst_ind))
    assert (Xt.index == inst_ind).all()
    assert (Xt.columns == inst_ind).all()
    np.testing.assert_allclose(Xt.to_numpy(), _expected_distmat(X))


@pytest.mark.skipif(
    not run_test_for_class(DistanceFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distance_features_hierarchical_repeated_instance_labels():
    """Test that series sharing a lower level index are not collapsed, see #8077."""
    # both hierarchy nodes contain a series labelled "s0", with different values
    index = pd.MultiIndex.from_product(
        [["a", "b"], ["s0"], pd.RangeIndex(3)], names=["h0", "h1", "time"]
    )
    X = pd.DataFrame({"c0": [0.0, 0.0, 0.0, 3.0, 4.0, 0.0]}, index=index)

    Xt = DistanceFeatures().fit_transform(X)

    assert Xt.shape == (2, 2)
    np.testing.assert_allclose(Xt.to_numpy(), [[0.0, 5.0], [5.0, 0.0]])


@pytest.mark.skipif(
    not run_test_for_class(DistanceFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distance_features_flatten_hierarchy():
    """Test that flatten_hierarchy flattens index and columns of the return."""
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), n_columns=1, min_timepoints=4, max_timepoints=4
    )

    Xt = DistanceFeatures(flatten_hierarchy=True).fit_transform(X)

    expected = ["h0_0__h1_0", "h0_0__h1_1", "h0_1__h1_0", "h0_1__h1_1"]
    assert list(Xt.index) == expected
    assert list(Xt.columns) == expected


@pytest.mark.skipif(
    not run_test_for_class(DistanceFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distance_features_hierarchical_fit_transform_different():
    """Test hierarchical data with different instances in fit and transform."""
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), n_columns=1, min_timepoints=4, max_timepoints=4
    )
    X_train = X.loc[["h0_0"]]
    X_test = X.loc[["h0_1"]]

    Xt = DistanceFeatures().fit(X_train).transform(X_test)

    assert Xt.shape == (2, 2)
    assert (Xt.index == X_test.index.droplevel(-1).unique()).all()
    assert (Xt.columns == X_train.index.droplevel(-1).unique()).all()
