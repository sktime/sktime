#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for index utilities."""

__author__ = ["mateenali66"]

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sktime.utils.index import fold_fingerprint


def _series(index):
    return pd.Series(range(len(index)), index=index)


def test_fingerprint_is_deterministic():
    """Same indices give the same fingerprint."""
    y = _series(pd.RangeIndex(10))
    assert fold_fingerprint(y[:6], y[6:]) == fold_fingerprint(y[:6], y[6:])


def test_fingerprint_ignores_values():
    """Fingerprint depends on the index only, not on the values."""
    index = pd.RangeIndex(10)
    a = pd.Series(np.arange(10), index=index)
    b = pd.Series(np.arange(10) * 3.5, index=index)
    assert fold_fingerprint(a) == fold_fingerprint(b)


def test_fingerprint_differs_for_different_folds():
    """Moving one element between train and test changes the fingerprint."""
    y = _series(pd.RangeIndex(10))
    assert fold_fingerprint(y[:6], y[6:]) != fold_fingerprint(y[:5], y[5:])


def test_fingerprint_is_position_sensitive():
    """Train and test are not interchangeable."""
    y = _series(pd.RangeIndex(10))
    assert fold_fingerprint(y[:6], y[6:]) != fold_fingerprint(y[6:], y[:6])


def test_fingerprint_is_order_sensitive():
    """Reordering an index changes the fingerprint."""
    assert fold_fingerprint(pd.Index([1, 2, 3])) != fold_fingerprint(
        pd.Index([3, 2, 1])
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(6),
        pd.Index(["a", "b", "c", "d", "e", "f"]),
        pd.date_range("2020-01-01", periods=6),
        pd.period_range("2020-01", periods=6, freq="M"),
        pd.MultiIndex.from_product([["h0", "h1"], range(3)]),
    ],
)
def test_fingerprint_supports_index_types(index):
    """Common index types are hashable and give a stable digest."""
    y = _series(index)
    fingerprint = fold_fingerprint(y)
    assert isinstance(fingerprint, str)
    assert fingerprint == fold_fingerprint(y)


def test_fingerprint_returns_none_without_index():
    """Objects with no index give None instead of raising."""
    assert fold_fingerprint(np.zeros((2, 2, 2))) is None
    assert fold_fingerprint(pd.Series(range(3)), object()) is None


def test_fingerprint_is_stable_across_processes():
    """Fingerprints do not depend on the per process hash seed.

    Regression test against use of the built-in ``hash``, which is salted per
    process for ``str`` and ``bytes``, and would make fingerprints from two
    ``evaluate`` runs incomparable.
    """
    script = (
        "import pandas as pd;"
        "from sktime.utils.index import fold_fingerprint;"
        "y = pd.Series(range(6), index=pd.Index(list('abcdef')));"
        "print(fold_fingerprint(y[:4], y[4:]))"
    )
    digests = []
    for seed in ["0", "1", "12345"]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        digests.append(out.stdout.strip())

    assert len(set(digests)) == 1
    y = _series(pd.Index(list("abcdef")))
    assert digests[0] == fold_fingerprint(y[:4], y[4:])
