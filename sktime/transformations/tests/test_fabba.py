"""Tests for the FABBA transformer."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.fabba import FABBA


def _make_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(np.cumsum(rng.standard_normal(n)))


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("sorting", ["norm", "pca", None])
def test_fabba_sorting_options(sorting):
    """All documented sorting options should run, and reconstruct the series."""
    X = _make_series()
    transformer = FABBA(
        tolerance=0.2,
        alpha=0.3,
        sorting=sorting,
        auto_digitize=False,
        n_jobs=1,
        return_start_values=True,
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) > 1
    assert len(Xt) < len(X)

    X_inv = transformer.inverse_transform(Xt)
    assert isinstance(X_inv, pd.DataFrame)
    assert len(X_inv) == len(X)
    # the reconstruction starts at the start value of the series
    assert X_inv.iloc[0, 0] == pytest.approx(X.iloc[0, 0])


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_aggregate_no_sorting_labels():
    """Aggregation without sorting assigns all points a label, in input order."""
    transformer = FABBA(sorting=None, alpha=0.5, auto_digitize=False)
    transformer._alpha = 0.5
    data = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.2, 5.1], [0.2, 0.0]])

    labels, starting_points = transformer._custom_aggregate(data)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 0])
    assert [sp[0] for sp in starting_points] == [0, 2]
    assert [sp[2] for sp in starting_points] == [3, 2]


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_aggregate_pca_single_feature():
    """PCA sorting with a single feature should sort by the centered feature."""
    transformer = FABBA(sorting="pca", alpha=0.5, auto_digitize=False)
    transformer._alpha = 0.5
    data = np.array([[3.0], [0.0], [3.2], [0.1], [10.0]])

    labels, starting_points = transformer._custom_aggregate(data)

    # points 1 and 3 are within alpha of each other, as are 0 and 2, 4 is alone
    assert labels[1] == labels[3]
    assert labels[0] == labels[2]
    assert len({labels[0], labels[1], labels[4]}) == 3
    assert len(starting_points) == 3


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("params", [{"auto_digitize": True}, {"alpha": None}])
def test_fabba_auto_digitize(params):
    """auto_digitize determines alpha from the data and uses it in aggregation."""
    X = _make_series()
    transformer = FABBA(tolerance=0.2, n_jobs=1, **params)
    Xt = transformer.fit_transform(X)

    assert len(Xt) > 1
    assert transformer._alpha is not None and transformer._alpha > 0

    # the automatically determined alpha is used, not the default alpha
    manual = FABBA(tolerance=0.2, n_jobs=1, alpha=transformer._alpha)
    pd.testing.assert_frame_equal(manual.fit_transform(X), Xt)
    assert manual._k == transformer._k


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_auto_digitize_panel():
    """auto_digitize works for a panel of several series."""
    X = [_make_series(n=100, seed=i) for i in range(3)]
    transformer = FABBA(tolerance=0.2, n_jobs=1, auto_digitize=True)
    Xt = transformer.fit_transform(X)

    # one symbolic series per instance, as a multi-indexed frame
    assert Xt.index.get_level_values(0).nunique() == 3
    assert transformer._alpha > 0
