def plot_time_series_with_change_points(*args, **kwargs):
    try:
        from sktime.detection.plotting.utils import (
            plot_time_series_with_change_points,
        )
    except ImportError as e:
        raise ImportError(
            "plot_time_series_with_change_points requires optional detection dependencies."
        ) from e

    return plot_time_series_with_change_points(*args, **kwargs)


def plot_time_series_with_profiles(*args, **kwargs):
    try:
        from sktime.detection.plotting.utils import (
            plot_time_series_with_profiles,
        )
    except ImportError as e:
        raise ImportError(
            "plot_time_series_with_profiles requires optional detection dependencies."
        ) from e

    return plot_time_series_with_profiles(*args, **kwargs)