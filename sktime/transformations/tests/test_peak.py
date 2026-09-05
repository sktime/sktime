# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for PeakTimeFeature."""

import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.peak import PeakTimeFeature
from sktime.utils._testing.hierarchical import _make_hierarchical


def _transformer():
    return PeakTimeFeature(
        ts_freq="D",
        peak_day_start=[1],
        peak_day_end=[3],
        keep_original_columns=True,
    )


@pytest.mark.skipif(
    not run_test_for_class(PeakTimeFeature),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("hierarchy_levels", [(3,), (2, 2)])
def test_peak_time_feature_multiindex(hierarchy_levels):
    """Test panel and hierarchical input, see issue #9010.

    ``PeakTimeFeature`` declared ``pd-multiindex`` and ``pd_multiindex_hier`` in
    ``X_inner_mtype``, but ``_transform`` infers the frequency from the index with
    ``pd.infer_freq``, which returns ``None`` for the interleaved dates of a panel.
    Indexing into that raised ``TypeError: 'NoneType' object is not subscriptable``.
    """
    X = _make_hierarchical(
        hierarchy_levels=hierarchy_levels,
        n_columns=1,
        min_timepoints=10,
        max_timepoints=10,
    )

    Xt = _transformer().fit_transform(X)

    assert Xt.index.equals(X.index)
    assert "is_peak_day" in Xt.columns


@pytest.mark.skipif(
    not run_test_for_class(PeakTimeFeature),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_peak_time_feature_multiindex_matches_by_instance():
    """Test that hierarchical output equals applying the transform per instance."""
    X = _make_hierarchical(
        hierarchy_levels=(2, 2),
        n_columns=1,
        min_timepoints=10,
        max_timepoints=10,
    )

    Xt = _transformer().fit_transform(X)

    expected = pd.concat(
        {
            key: _transformer().fit_transform(group.droplevel([0, 1]))
            for key, group in X.groupby(level=[0, 1])
        },
        names=X.index.names[:2],
    )
    expected = expected.reorder_levels(Xt.index.names).sort_index()

    pd.testing.assert_frame_equal(Xt, expected)
