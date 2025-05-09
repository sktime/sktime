"""Tests for the kinematic transformers."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.kinematic import KinematicFeatures


@pytest.mark.skipif(
    not run_test_for_class(KinematicFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_kinematic_expected_output():
    """Test expected output in docstring example."""
    traj3d = pd.DataFrame(columns=["x", "y", "z"])
    traj3d["x"] = pd.Series(np.sin(np.arange(200) / 100))
    traj3d["y"] = pd.Series(np.cos(np.arange(200) / 100))
    traj3d["z"] = pd.Series(np.arange(200) / 100)

    t = KinematicFeatures()
    Xt = t.fit_transform(traj3d)

    assert np.allclose(np.sqrt(2) / 100, Xt["v_abs"][1:])
    assert np.allclose(1e-4, Xt["a_abs"][2:])
    assert np.allclose(0.5, Xt["curv"][2:], atol=1e-5)
