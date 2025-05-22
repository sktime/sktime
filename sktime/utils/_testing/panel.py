"""Utility functions for generating panel data and learning task scenarios."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = [
    "make_classification_problem",
    "make_regression_problem",
    "make_transformer_problem",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.datatypes import convert


def _make_panel(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    y=None,
    all_positive=False,
    random_state=None,
    return_mtype="pd-multiindex",
):
    """Generate sktime compatible test data, Panel data formats.

    Parameters
    ----------
    n_instances : int, optional, default=20
        number of instances per series in the panel
    n_columns : int, optional, default=1
        number of variables in the time series
    n_timepoints : int, optional, default=20
        number of time points in each series
    y : None (default), or 1D np.darray or 1D array-like, shape (n_instances, )
        if passed, return will be generated with association to y
    all_positive : bool, optional, default=False
        whether series contain only positive values when generated
    random_state : None (default) or int
        if int is passed, will be used in numpy RandomState for generation
    return_mtype : str, sktime Panel mtype str, default="pd-multiindex"
        see sktime.datatypes.MTYPE_LIST_PANEL for a full list of admissible strings
        see sktime.datatypes.MTYPE_REGISTER for an short explanation of formats
        see examples/AA_datatypes_and_datasets.ipynb for a full specification

    Returns
    -------
    X : an sktime time series data container of mtype return_mtype
        with n_instances instances, n_columns variables, n_timepoints time points
        generating distribution is all values i.i.d. normal with std 0.5
        if y is passed, i-th series values are additively shifted by y[i] * 100
    """
    # If target variable y is given, we ignore n_instances and instead generate as
    # many instances as in the target variable
    if y is not None:
        y = np.asarray(y)
        n_instances = len(y)
    rng = check_random_state(random_state)

    # Generate data as 3d numpy array
    X = rng.normal(scale=0.5, size=(n_instances, n_columns, n_timepoints))

    # Generate association between data and target variable
    if y is not None:
        X = X + (y * 100).reshape(-1, 1, 1)

    if all_positive:
        X = X**2

    X = convert(X, from_type="numpy3D", to_type=return_mtype)
    return X


def _make_panel_X(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    y=None,
    all_positive=False,
    return_numpy=False,
    random_state=None,
):
    if return_numpy:
        return_mtype = "numpy3D"
    else:
        return_mtype = "nested_univ"

    return _make_panel(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        y=y,
        all_positive=all_positive,
        random_state=random_state,
        return_mtype=return_mtype,
    )


def _make_regression_y(n_instances=20, return_numpy=True, random_state=None):
    rng = check_random_state(random_state)
    y = rng.normal(size=n_instances)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_classification_y(
    n_instances=20, n_classes=2, return_numpy=True, random_state=None
):
    if not n_instances > n_classes:
        raise ValueError("n_instances must be bigger than n_classes")
    rng = check_random_state(random_state)
    n_repeats = int(np.ceil(n_instances / n_classes))
    y = np.tile(np.arange(n_classes), n_repeats)[:n_instances]
    rng.shuffle(y)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def make_classification_problem(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    n_classes=2,
    return_numpy=False,
    random_state=None,
):
    """Make Classification Problem."""
    y = _make_classification_y(
        n_instances, n_classes, return_numpy=return_numpy, random_state=random_state
    )
    X = _make_panel_X(
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
        random_state=random_state,
        y=y,
    )

    return X, y


def make_regression_problem(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    return_numpy=False,
    random_state=None,
):
    """Make Regression Problem."""
    y = _make_regression_y(
        n_instances, random_state=random_state, return_numpy=return_numpy
    )
    X = _make_panel_X(
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
        random_state=random_state,
        y=y,
    )
    return X, y


def make_clustering_problem(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    return_numpy=False,
    random_state=None,
):
    """Make Clustering Problem."""
    # Can only currently support univariate so converting
    # to univaritate for the time being
    return _make_panel_X(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
        random_state=random_state,
    )


def make_transformer_problem(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    return_numpy=True,
    random_state=None,
    panel=True,
):
    """Make Transformer Problem."""
    if not panel:
        X = make_transformer_problem(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            return_numpy=True,
            random_state=random_state,
            panel=True,
        )
        if return_numpy:
            X = X[0]
        else:
            X = pd.DataFrame(X[0])
    else:
        X = _make_panel_X(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            return_numpy=True,
            random_state=random_state,
        )
        if not return_numpy:
            arr = []
            for data in X:
                arr.append(pd.DataFrame(data))
            X = arr

    return X


def _make_nested_from_array(array, n_instances=20, n_columns=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_columns)] for _ in range(n_instances)],
        columns=[f"col{c}" for c in range(n_columns)],
    )
