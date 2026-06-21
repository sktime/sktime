#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SCINetForecaster.

Pins the fix for sktime/sktime#10054: ``SCINetForecaster._build_network``
used to pass ``num_stacks=self.hid_size`` (a copy-paste typo), so the
user-supplied ``num_stacks`` was silently ignored. The underlying
``SCINet`` only constructs the second encoder block (``blocks2``) when
``num_stacks == 2``, so the bug also disabled stacking entirely whenever
``hid_size`` happened to be 1 (the default).
"""

__author__ = ["jbbqqf"]

import pytest

from sktime.forecasting.scinet import SCINetForecaster
from sktime.utils.dependencies import _check_soft_dependencies


# SCINet is in the EXCLUDE_ESTIMATORS list (known unrelated bug #7871) so the
# usual ``run_test_for_class`` switch would skip this. The fix tested here is
# narrow and deterministic — it only inspects the network builder — so we gate
# only on the torch soft-dep being present.
@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="torch soft dependency missing",
)
def test_scinet_num_stacks_propagated_to_underlying_network():
    """``num_stacks`` must reach the underlying ``SCINet`` unchanged.

    Builds the network through the forecaster's own
    ``_build_network`` helper with ``num_stacks=2`` and ``hid_size=1`` —
    the exact combination that exposes the bug, since on ``main`` both
    arguments would be set to ``hid_size=1`` and the second encoder block
    would never be built.
    """
    import numpy as np
    import pandas as pd

    forecaster = SCINetForecaster(
        seq_len=8,
        hid_size=1,
        num_stacks=2,
        num_levels=1,
        num_decoder_layer=1,
    )
    # ``_build_network`` reads ``self._y.shape[-1]`` to pick the input dim,
    # so we attach a tiny univariate series before invoking it directly.
    forecaster._y = pd.DataFrame({"y": np.arange(8, dtype=float)})

    network = forecaster._build_network(fh=2)

    # On ``origin/main`` this attribute would be ``1`` (the buggy value of
    # ``self.hid_size``); on the fix branch it must equal the
    # user-requested ``num_stacks``.
    assert network.stacks == 2, (
        f"SCINet was built with stacks={network.stacks}, expected 2. "
        "This indicates the num_stacks argument was overridden by "
        "hid_size — see sktime/sktime#10054."
    )
    # The second encoder block only exists when ``num_stacks == 2``; its
    # presence is the observable side-effect that proves the fix is wired
    # all the way through.
    assert hasattr(network, "blocks2"), (
        "SCINet.blocks2 was not built even though num_stacks=2 was "
        "requested — the constructor branch at sktime/networks/scinet.py "
        "around line 752 was not reached."
    )
