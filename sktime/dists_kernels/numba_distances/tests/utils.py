# -*- coding: utf-8 -*-
from sktime.datatypes import convert_to
from sktime.utils._testing.series import _make_series
from sktime.utils._testing.panel import _make_panel_X


def create_test_distance_numpy(
    n_instance: int,
    n_columns: int,
    n_timepoints: int,
    random_state: int = 1,
):
    """Method to create a test numpy distance.

    Parameters
    ----------
    n_instance: int
        Number of instances to create.
    n_columns: int
        Number of columns to create.
    n_timepoints: int
        Number of timepoints to create.
    random_state: int, defaults = 1
        Random state to initialise with.

    Returns
    -------
    np.ndarray
        Numpy array containing distance(s) specified.
    """
    df = _create_test_distances(
        n_instance=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    if n_instance > 1:
        return convert_to(df, to_type="numpy3D")
    else:
        return convert_to(df, to_type="np.ndarray")


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
