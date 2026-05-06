"""Tests for scipy interface."""
import warnings

import numpy as np
import pytest

from sktime.dists_kernels.scipy_dist import ScipyDist
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_transformer_problem


@pytest.fixture
def X1():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=1,
        return_numpy=True,
        panel=False,
    )


@pytest.fixture
def X2():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=2,
        return_numpy=True,
        panel=False,
    )


@pytest.fixture
def X1_df():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=1,
        return_numpy=False,
        panel=False,
    )


@pytest.fixture
def X2_df():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=2,
        return_numpy=False,
        panel=False,
    )

def _scipy_available(metric):
    """
    Return
        True if scipy accepts metric
        False if not
    """
    from scipy.spatial.distance import cdist

    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cdist(x, x, metric=metric)
    except ValueError:
        return False

    return True


def _get_kul_name():
    """Get name of kul... distance.

    Utility to bridge deprecation of kulsinski distance in scipy.
    Name pre-1.11.0 is kulsinski, and from 1.11.0 it is kulczynski1.
    From 1.17.0 none of them is available returns None in that case.
    """
    distance = ["kulczynski1", "kulsinski"]
    for name in distance:
        if _scipy_available(name):
            return name

    return None


# potential parameters
_METRIC_CANDIDATES = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    *filter(None, [_get_kul_name()]),
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
METRIC_VALUES = [
    metric for metric in _METRIC_CANDIDATES if _scipy_available(metric)
]

P_VALUES = [1, 2, 5, 10]
COLALIGN_VALUES = ["intersect", "force-align", "none"]


@pytest.mark.skipif(
    not run_test_for_class(ScipyDist),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scipydist(X1, X2, X1_df, X2_df):
    """Test runner for numpy and dataframe tests."""
    # test numpy
    _run_scipy_dist_test(X1, X2)

    # test dataframe
    _run_scipy_dist_test(X1_df, X2_df)


def _run_scipy_dist_test(x, y):
    # default parameters
    default_params = ScipyDist()
    default_params_transformation = np.around(
        (default_params.transform(x, y)), decimals=3
    )
    assert np.array_equal(
        np.array(
            [
                [2.318, 1.657, 1.582, 1.502, 1.461],
                [1.79, 1.249, 1.715, 1.656, 1.449],
                [2.424, 2.083, 2.28, 1.735, 1.73],
                [1.602, 1.012, 1.658, 1.167, 0.901],
                [2.219, 1.643, 1.373, 1.005, 1.216],
            ]
        ),
        default_params_transformation,
    ), "Error occurred testing on default parameters, result is not correct"

    for metric in METRIC_VALUES:
        for p in P_VALUES:
            for colalign in COLALIGN_VALUES:
                metric_params = ScipyDist(metric=metric, p=p, colalign=colalign)
                metric_params_transformation = metric_params.transform(x, y)
                assert isinstance(metric_params_transformation, np.ndarray), (
                    f"Error occurred testing on following parameters"
                    f"metric={metric}, p={p}, colalign={colalign}"
                )
