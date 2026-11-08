"""Tests for the kinematic transformers."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.kinematic import KinematicFeatures


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


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
@pytest.mark.parametrize("n_dimensions", [1, 3, 8, 20])
@pytest.mark.parametrize("index_type", ["range", "datetime"])
def test_kinematic_row_reductions(dtype, n_dimensions, index_type):
    """Match individual Series reductions, including their floating-point order."""
    values = np.random.default_rng(11375).normal(size=(30, n_dimensions))
    X = pd.DataFrame((values * 10).astype(dtype).cumsum(axis=0))
    if index_type == "datetime":
        X.index = pd.date_range("2020-01-01", periods=len(X), name="time")

    velocity = X.diff()
    acceleration = velocity.diff()

    def squared_norms(frame):
        return pd.Series([(row**2).sum() for _, row in frame.iterrows()], index=X.index)

    vsq = squared_norms(velocity)
    asq = squared_norms(acceleration)
    cross_term = (velocity.to_numpy() * acceleration.to_numpy()).sum(axis=1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        curvature = np.sqrt(
            np.abs((vsq.to_numpy() * asq.to_numpy() - cross_term) / vsq.to_numpy() ** 3)
        )
    expected = pd.DataFrame(
        {"v_abs": vsq**0.5, "a_abs": asq**0.5, "curv": curvature}, index=X.index
    )
    expected.iloc[0, 0] = np.nan
    expected.iloc[:2, 1] = np.nan

    actual = KinematicFeatures().fit_transform(X)
    assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_kinematic_stationary_trajectory(dtype):
    """Preserve zero speed/acceleration and undefined curvature for stationary data."""
    X = pd.DataFrame(np.ones((5, 3), dtype=dtype))
    expected = pd.DataFrame(
        {
            "v_abs": [np.nan, 0, 0, 0, 0],
            "a_abs": [np.nan, np.nan, 0, 0, 0],
            "curv": [np.nan] * 5,
        },
        dtype=dtype,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        actual = KinematicFeatures().fit_transform(X)
    assert_frame_equal(actual, expected, check_exact=True)
