"""Tests for the distance measure getters of the ProximityForest family.

The msm, erp and lcss getters sample the dimension to compare via a
``dim_to_use`` parameter. These tests pin that the sampling range tracks the
number of channels in the training data, and that the returned distance
measures actually restrict the computation to the sampled dimension.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.classification.distance_based import ProximityStump
from sktime.classification.distance_based._proximity_forest import (
    erp_distance_measure_getter,
    lcss_distance_measure_getter,
    msm_distance_measure_getter,
)

GETTERS = [
    (msm_distance_measure_getter, {"c": 0.5}),
    (erp_distance_measure_getter, {"g": 0.1, "band_size": 2}),
    (lcss_distance_measure_getter, {"epsilon": 0.1, "delta": 2}),
]


def _nested_dataframe(n_cases, n_channels, n_timepoints, seed=42):
    rng = np.random.default_rng(seed)
    data = {
        channel: [pd.Series(rng.random(n_timepoints)) for _ in range(n_cases)]
        for channel in range(n_channels)
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("getter, params", GETTERS)
def test_dim_to_use_range_tracks_number_of_channels(getter, params):
    X_univariate = _nested_dataframe(5, 1, 12)
    X_multivariate = _nested_dataframe(5, 3, 12)

    # scipy randint is exclusive on the upper bound, so support is (0, high - 1)
    assert getter(X_univariate)["dim_to_use"].support() == (0, 0)
    assert getter(X_multivariate)["dim_to_use"].support() == (0, 2)


@pytest.mark.parametrize("getter, params", GETTERS)
def test_distance_measure_restricted_to_sampled_dimension(getter, params):
    from sktime.dists_kernels._numba_distances import (
        erp_distance,
        lcss_distance,
        msm_distance,
    )

    direct_distances = {
        id(msm_distance_measure_getter): msm_distance,
        id(erp_distance_measure_getter): erp_distance,
        id(lcss_distance_measure_getter): lcss_distance,
    }
    distance_fn = direct_distances[id(getter)]

    X = _nested_dataframe(2, 3, 12)
    instance_a = X.iloc[0, :]
    instance_b = X.iloc[1, :]

    measure = getter(X)["distance_measure"][0]
    for dim in (0, 1, 2):
        expected = distance_fn(
            np.asarray(instance_a.iloc[dim], dtype=np.float64),
            np.asarray(instance_b.iloc[dim], dtype=np.float64),
            **params,
        )
        result = measure(instance_a, instance_b, dim_to_use=dim, **params)
        assert result == pytest.approx(expected)


@pytest.mark.parametrize("distance_measure", ["msm", "erp", "lcss"])
def test_proximity_stump_univariate_fit_predict(distance_measure):
    X = _nested_dataframe(30, 1, 20, seed=0)
    y = np.array(["a", "b"])[np.random.default_rng(1).integers(0, 2, 30)]

    clf = ProximityStump(distance_measure=distance_measure, random_state=0, n_jobs=1)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == 30
    assert set(preds).issubset({"a", "b"})
