import numpy as np
import pandas as pd
import pytest

from sktime.detection.plotting.utils import (
    plot_time_series_with_change_points,
    plot_time_series_with_profiles,
)
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.fixture
def time_series_data():
    ts_data = np.random.rand(100)
    ts = pd.DataFrame({"Data": ts_data})
    true_cps = [4, 8]
    font_size = 12
    ts_name = "Test Time Series"
    profiles = np.array([np.random.rand(100) for _ in range(20)])
    found_cps = [35, 65]
    score_name = "Custom Score"
    return {
        "ts_name": ts_name,
        "ts": ts,
        "true_cps": true_cps,
        "font_size": font_size,
        "profiles": profiles,
        "found_cps": found_cps,
        "score_name": score_name,
    }


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", "seaborn", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_change_points(time_series_data):
    import matplotlib.pyplot as plt

    # Access data from the fixture
    ts_name = time_series_data["ts_name"]
    ts = time_series_data["ts"]
    true_cps = time_series_data["true_cps"]
    font_size = time_series_data["font_size"]

    fig, ax = plot_time_series_with_change_points(ts_name, ts, true_cps, font_size)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == ts_name


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_profiles(time_series_data):
    import matplotlib.pyplot as plt

    # Access data from the fixture
    ts_name = time_series_data["ts_name"]
    ts = time_series_data["ts"]
    true_cps = time_series_data["true_cps"]
    font_size = time_series_data["font_size"]
    profiles = time_series_data["profiles"]
    found_cps = time_series_data["found_cps"]
    score_name = time_series_data["score_name"]

    fig, ax = plot_time_series_with_profiles(
        ts_name, ts, profiles, true_cps, found_cps, score_name, font_size
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert ax[0].get_title() == ts_name
