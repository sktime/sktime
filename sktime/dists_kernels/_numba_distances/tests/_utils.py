import time
from collections.abc import Callable

from sktime.datatypes import convert_to
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils._testing.series import _make_series


def create_test_distance_numpy(
    n_instance: int,
    n_columns: int = None,
    n_timepoints: int = None,
    random_state: int = 1,
):
    """Create a test numpy distance.

    Parameters
    ----------
    n_instance: int
        Number of instances to create.
    n_columns: int
        Number of columns to create.
    n_timepoints: int, defaults = None
        Number of timepoints to create in each column.
    random_state: int, defaults = 1
        Random state to initialise with.

    Returns
    -------
    np.ndarray 2D or 3D numpy
        Numpy array of shape specific. If 1 instance then 2D array returned,
        if > 1 instance then 3D array returned.
    """
    num_dims = 3
    if n_timepoints is None:
        n_timepoints = 1
        num_dims -= 1
    if n_columns is None:
        n_columns = 1
        num_dims -= 1

    df = _create_test_distances(
        n_instance=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    if num_dims == 3:
        return convert_to(df, to_type="numpy3D")
    elif num_dims == 2:
        return convert_to(df, to_type="numpy3D")[:, :, 0]
    else:
        return convert_to(df, to_type="numpy3D")[:, 0, 0]


def _create_test_distances(n_instance, n_columns, n_timepoints, random_state=1):
    if n_instance > 1:
        return _make_panel_X(
            n_instances=n_instance,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            random_state=random_state,
        )
    else:
        return _make_series(n_timepoints, n_columns, random_state=random_state)


def _time_distance(callable: Callable, average: int = 30, **kwargs):
    for _ in range(3):
        callable(**kwargs)

    total = 0
    for _ in range(average):
        start = time.time()
        callable(**kwargs)
        total += time.time() - start

    return total / average
