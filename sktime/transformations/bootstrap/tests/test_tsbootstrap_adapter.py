#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TSBootstrapAdapter with vendored tsbootstrap classes."""

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.libs.tsbootstrap import BlockBootstrap
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.bootstrap import TSBootstrapAdapter


@pytest.mark.skipif(
    not run_test_for_class(TSBootstrapAdapter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsbootstrap_adapter_with_vendored_blockbootstrap_returns_index_col():
    """Vendored BlockBootstrap should work through TSBootstrapAdapter."""
    y = load_airline().to_frame()

    adapter = TSBootstrapAdapter(
        bootstrap=BlockBootstrap(n_bootstraps=2, block_length=4),
        return_indices=True,
    )
    Xt = adapter.fit_transform(y)

    assert isinstance(Xt, pd.DataFrame)
    assert "resampled_index" in Xt.columns
    assert Xt.index.get_level_values(0).nunique() == 2
