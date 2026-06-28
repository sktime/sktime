"""Tests for the EvoForestTSWM transformer."""

__author__ = ["kayuksel"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.evoforest_tswm import EvoForestTSWM


def _fixture_X():
    rng = np.random.RandomState(42)
    base = np.sin(np.linspace(0, 6, 80))[None, None, :]
    return base * np.array([1, 2, 3])[:, None, None] + 0.1 * rng.normal(size=(3, 1, 80))


# frozen aggregates of the discovered champion on _fixture_X() (torch-free)
_EXPECT = {
    "full": dict(
        shape=(3, 423),
        agg=[968.166837, 0.762937, 2.12074, 0.030663, 0.578125, -8.210593, 13.586488],
    ),
    "pruned": dict(
        shape=(3, 245),
        agg=[826.224867, 1.124115, 2.243575, 0.211374, 0.070312, -8.210593, 9.864845],
    ),
}


def _agg(a):
    a = np.asarray(a)
    return [
        float(a.sum()),
        float(a.mean()),
        float(a.std()),
        float(a[0, 0]),
        float(a[-1, -1]),
        float(a.min()),
        float(a.max()),
    ]


@pytest.mark.skipif(
    not run_test_for_class(EvoForestTSWM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("pooling", ["full", "pruned"])
def test_evoforest_tswm_shape_and_values(pooling):
    """Output shape and a frozen numeric regression lock the discovered encoder."""
    out = EvoForestTSWM(pooling=pooling).fit_transform(_fixture_X())
    exp = _EXPECT[pooling]
    assert isinstance(out, pd.DataFrame)
    assert out.shape == exp["shape"]
    np.testing.assert_allclose(_agg(out), exp["agg"], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not run_test_for_class(EvoForestTSWM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_evoforest_tswm_multivariate():
    """Output width is independent of the channel count."""
    rng = np.random.RandomState(0)
    out = EvoForestTSWM().fit_transform(rng.normal(size=(4, 3, 70)))
    assert out.shape == (4, 423)
