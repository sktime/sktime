# -*- coding: utf-8 -*-

from sktime.utils._testing.series import _make_series


def make_annotation_problem(
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=2,
    random_state=None,
):
    y = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )

    if not make_X:
        return y

    X = _make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )
    X.index = y.index
    return y, X
