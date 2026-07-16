"""Tests for CiscoTSMForecaster probabilistic forecasts (#10538)."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.cisco_tsm import CiscoTSMForecaster

_HAS_HISTOGRAM_QPD = False
if _check_soft_dependencies("skpro", severity="none"):
    try:
        from skpro.distributions import HistogramQPD  # noqa: F401

        _HAS_HISTOGRAM_QPD = True
    except ImportError:
        pass


def _fit_dummy(**kwargs):
    y = pd.Series(np.arange(30, dtype=float), index=pd.RangeIndex(30), name="y")
    f = CiscoTSMForecaster(ignore_deps=True, **kwargs)
    with patch("sktime.forecasting.base._base._check_estimator_deps"):
        f.fit(y)
    return f


@pytest.mark.skipif(not _HAS_HISTOGRAM_QPD, reason="skpro>=2.14 with HistogramQPD")
def test_predict_proba_is_histogram_qpd_not_normal():
    """#10538: no Normal-from-IQR default."""
    from skpro.distributions import HistogramQPD
    from skpro.distributions.normal import Normal

    dist = _fit_dummy().predict_proba(fh=[1, 2, 3])
    assert isinstance(dist, HistogramQPD)
    assert not isinstance(dist, Normal)


@pytest.mark.skipif(not _HAS_HISTOGRAM_QPD, reason="skpro>=2.14 with HistogramQPD")
def test_predict_proba_matches_predict_quantiles():
    """proba quantiles match predict_quantiles (knot, mid, clamp)."""
    f = _fit_dummy()
    fh = [1, 2, 3]
    alphas = [0.1, 0.33, 0.001]  # native, interpolated, out-of-range clamp

    q_table = f.predict_quantiles(fh=fh, alpha=alphas)
    q_dist = f.predict_proba(fh=fh).quantile(alpha=alphas)

    assert q_table.columns.equals(q_dist.columns)
    assert q_table.index.equals(q_dist.index)
    np.testing.assert_allclose(
        q_table.to_numpy(dtype=float),
        q_dist.to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not _HAS_HISTOGRAM_QPD, reason="skpro>=2.14 with HistogramQPD")
def test_predict_interval_matches_quantiles():
    """predict_interval ends equal the matching quantile pair."""
    f = _fit_dummy()
    fh = [1, 2, 3]
    coverage = 0.8
    lo_a, hi_a = 0.1, 0.9

    iv = f.predict_interval(fh=fh, coverage=coverage)
    q = f.predict_quantiles(fh=fh, alpha=[lo_a, hi_a])

    np.testing.assert_allclose(
        iv[("y", coverage, "lower")].to_numpy(dtype=float),
        q[("y", lo_a)].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        iv[("y", coverage, "upper")].to_numpy(dtype=float),
        q[("y", hi_a)].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not _HAS_HISTOGRAM_QPD, reason="skpro>=2.14 with HistogramQPD")
def test_pickle_roundtrip_predict_proba():
    """Unpickle reloads model; proba still HistogramQPD and matches quantiles."""
    import pickle

    from skpro.distributions import HistogramQPD

    f = _fit_dummy()
    f2 = pickle.loads(pickle.dumps(f))
    dist = f2.predict_proba(fh=[1, 2])
    assert isinstance(dist, HistogramQPD)

    alphas = [0.1, 0.9]
    np.testing.assert_allclose(
        f.predict_quantiles(fh=[1, 2], alpha=alphas).to_numpy(dtype=float),
        f2.predict_quantiles(fh=[1, 2], alpha=alphas).to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )
