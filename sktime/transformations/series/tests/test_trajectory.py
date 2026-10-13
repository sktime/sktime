# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for trajectory transformers."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.trajectory import DouglasPeuckerTrajectoryGeneralizer
from sktime.utils.dependencies import _check_soft_dependencies


def _make_trajectory(n=50):
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "lat": np.linspace(0, 1, n) + rng.normal(0, 0.005, n),
            "lon": np.linspace(0, 1, n) + rng.normal(0, 0.005, n),
        },
        index=dates,
    )
    return df


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "movingpandas", "geopandas", "shapely", severity="none"
    ),
    reason="movingpandas, geopandas, shapely not installed",
)
@pytest.mark.skipif(
    not run_test_for_class(DouglasPeuckerTrajectoryGeneralizer),
    reason="run_test_for_class returned False, skipping per testing policy",
)
def test_dp_trajectory_generalizer_simplifies():
    """Output has fewer or equal points than input."""
    X = _make_trajectory(100)
    transformer = DouglasPeuckerTrajectoryGeneralizer(tolerance=0.1)
    Xt = transformer.fit_transform(X)
    assert len(Xt) <= len(X)
    assert "lat" in Xt.columns and "lon" in Xt.columns


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "movingpandas", "geopandas", "shapely", severity="none"
    ),
    reason="movingpandas, geopandas, shapely not installed",
)
@pytest.mark.skipif(
    not run_test_for_class(DouglasPeuckerTrajectoryGeneralizer),
    reason="run_test_for_class returned False, skipping per testing policy",
)
def test_dp_trajectory_generalizer_preserves_endpoints():
    """First and last points should be preserved."""
    X = _make_trajectory(50)
    transformer = DouglasPeuckerTrajectoryGeneralizer(tolerance=0.001)
    Xt = transformer.fit_transform(X)
    assert np.isclose(Xt["lat"].iloc[0], X["lat"].iloc[0], atol=1e-6)
    assert np.isclose(Xt["lat"].iloc[-1], X["lat"].iloc[-1], atol=1e-6)


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "movingpandas", "geopandas", "shapely", severity="none"
    ),
    reason="movingpandas, geopandas, shapely not installed",
)
@pytest.mark.skipif(
    not run_test_for_class(DouglasPeuckerTrajectoryGeneralizer),
    reason="run_test_for_class returned False, skipping per testing policy",
)
def test_dp_trajectory_generalizer_higher_tolerance_more_aggressive():
    """Higher tolerance should produce fewer points."""
    X = _make_trajectory(200)
    low = DouglasPeuckerTrajectoryGeneralizer(tolerance=0.001).fit_transform(X)
    high = DouglasPeuckerTrajectoryGeneralizer(tolerance=0.5).fit_transform(X)
    assert len(high) <= len(low)
