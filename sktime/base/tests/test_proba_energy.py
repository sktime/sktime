# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Monte Carlo default of ``BaseDistribution.energy``."""

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.base._proba._base import BaseDistribution
from sktime.base._proba._normal import Normal
from sktime.tests.test_switch import run_test_module_changed


def _energy_rowwise(dist, x=None):
    """Reference for the energy default, computed row by row with ``apply``."""
    N = dist.get_tag("approx_energy_spl")
    if x is None:
        splx = dist.sample(N)
    else:
        splx = pd.concat([x] * N, keys=range(N))
    sply = dist.sample(N)
    energy = (splx - sply).apply(np.linalg.norm, axis=1, ord=1)
    energy = energy.groupby(level=1, sort=False).mean()
    return pd.DataFrame(energy, index=dist.index, columns=["energy"])


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base"),
    reason="Run if base module has changed.",
)
@pytest.mark.parametrize("x_kind", [None, "values", "values_with_nan"])
def test_energy_default_matches_rowwise(x_kind):
    """Check the energy default against a row-wise reference on the same samples.

    Also checks that NaN in ``x`` propagates to the energy of that row,
    and that the approximation is close to the exact energy of ``Normal``.
    """
    dist = Normal(
        mu=[[0, 1], [2, 3], [4, 5]],
        sigma=[[1, 2], [1, 1], [0.5, 3]],
        index=pd.Index([1, 2, 5]),
        columns=pd.Index(["a", "b"]),
    )

    if x_kind is None:
        x = None
    else:
        x = pd.DataFrame(
            [[0.5, 1.0], [2.0, -1.0], [4.0, 6.0]],
            index=dist.index,
            columns=dist.columns,
        )
        if x_kind == "values_with_nan":
            x.iloc[1, 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(42)
        expected = _energy_rowwise(dist, x)
        np.random.seed(42)
        actual = BaseDistribution.energy(dist, x)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    if x_kind == "values_with_nan":
        assert actual.isna().to_numpy().ravel().tolist() == [False, True, False]
    else:
        exact = dist.energy(x)
        np.testing.assert_allclose(actual.to_numpy(), exact.to_numpy(), rtol=0.2)
