import numpy as np
import pandas as pd
import pytest

from sktime.annotation.plotting.utils import (
    plot_time_series_with_anomalies,
    plot_time_series_with_change_point_detection,
    plot_time_series_with_change_points,
    plot_time_series_with_detrender,
    plot_time_series_with_predicted_anomalies,
    plot_time_series_with_profiles,
    plot_time_series_with_subsequent_outliers,
)
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.fixture
def time_series_data():
    ts_data = np.random.rand(100)
    labels = np.random.choice([0.0, 1.0], size=100)
    ts = pd.DataFrame({"Data": ts_data, "label": labels})
    intervals = [(5, 10), (15, 20)]
    y_hat = pd.Series(np.random.choice([True, False], size=100))
    true_cps = [4, 8]
    font_size = 12
    ts_name = "Test Time Series"
    profiles = np.array([np.random.rand(100) for _ in range(20)])
    found_cps = [35, 65]
    score_name = "Custom Score"
    return {
        "ts_name": ts_name,
        "ts": ts,
        "intervals": intervals,
        "y_hat": y_hat,
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


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_anomalies(time_series_data):
    import matplotlib.axes as axes
    import matplotlib.pyplot as plt

    # Accesing data from the fixture
    df = time_series_data["ts"]
    X = df.loc[df.iloc[:, 1] == 1.0].index
    y = df.loc[df.iloc[:, 1] == 1.0, df.columns[0]]

    fig, ax = plot_time_series_with_anomalies(df, X, y)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, axes.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_predicted_anomalies(time_series_data):
    import matplotlib.axes as axes
    import matplotlib.pyplot as plt

    df = time_series_data["ts"]
    y_hat = time_series_data["y_hat"]

    fig, ax = plot_time_series_with_predicted_anomalies(df, y_hat)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray) and all(isinstance(a, axes.Axes) for a in ax)


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_detrender(time_series_data):
    import matplotlib.axes as axes
    import matplotlib.pyplot as plt

    df = time_series_data["ts"]
    y_hat = time_series_data["y_hat"]

    fig, ax = plot_time_series_with_detrender(df, y_hat)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray) and all(isinstance(a, axes.Axes) for a in ax)


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_subsequent_outliers(time_series_data):
    import matplotlib.axes as axes
    import matplotlib.pyplot as plt

    df = time_series_data["ts"]
    intervals = [
        pd.Interval(left, right) for left, right in time_series_data["intervals"]
    ]
    fig, ax = plot_time_series_with_subsequent_outliers(df, intervals)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, axes.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies("seaborn", "matplotlib", severity="none"),
    reason="Seaborn is required as a dependency for this plot.",
)
def test_plot_time_series_with_change_point_detection(time_series_data):
    import matplotlib.axes as axes
    import matplotlib.pyplot as plt

    df = time_series_data["ts"]
    true_cps = time_series_data["true_cps"]
    found_cps = time_series_data["found_cps"]
    predicted_change_points = pd.DataFrame({"Predicted Change Points": found_cps})

    fig, ax = plot_time_series_with_change_point_detection(
        df, true_cps[0], predicted_change_points
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, axes.Axes)
