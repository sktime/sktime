__author__ = ["fkiraly", "klam-data"]
__all__ = []

from sklearn.utils import check_random_state

from sktime.detection.datagen import piecewise_poisson
from sktime.utils._testing.series import _make_series


def make_detection_problem(
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=2,
    random_state=None,
    estimator_type=None,
):
    if estimator_type == "Poisson":
        if make_X:
            raise ValueError("PoissonHMM creates a distribution for y only")
        rng = check_random_state(random_state)
        y = piecewise_poisson(
            lambdas=rng.randint(1, 4, n_timepoints),
            lengths=rng.randint(1, 5, n_timepoints),
            random_state=random_state,
        )
        return y

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
