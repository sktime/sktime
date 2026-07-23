# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SundialForecaster probabilistic API."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.sundial import SundialForecaster

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("torch", "transformers", "skpro", severity="none"),
    reason="torch, transformers, and skpro required for Sundial probabilistic tests",
)


def _make_y(n_timepoints=10, n_columns=1):
    index = pd.RangeIndex(n_timepoints)
    data = np.arange(n_timepoints, dtype=float).reshape(-1, 1)
    if n_columns > 1:
        data = np.column_stack([data * (i + 1) for i in range(n_columns)])
    cols = [f"col_{i}" for i in range(n_columns)]
    return pd.DataFrame(data, index=index, columns=cols)


@pytest.mark.parametrize("n_columns", [1, 2])
def test_sundial_predict_proba_empirical_consistency(n_columns):
    """Sundial probabilistic API returns Empirical distribution and valid quantiles."""
    from skpro.distributions.empirical import Empirical

    params = SundialForecaster.get_test_params()[0]
    forecaster = SundialForecaster(**params)
    y = _make_y(n_timepoints=10, n_columns=n_columns)
    fh = [1, 2, 3]
    sparse_fh = [1, 3]
    alpha = [0.1, 0.5, 0.9]

    forecaster.fit(y, fh=fh)

    pred_dist = forecaster.predict_proba(fh=fh)
    assert isinstance(pred_dist, Empirical)

    expected_index = forecaster.fh.to_absolute_index(forecaster.cutoff)
    assert pred_dist.index.equals(expected_index)
    assert list(pred_dist.columns) == list(y.columns)

    quantiles_from_predict = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    assert quantiles_from_predict.index.equals(expected_index)
    assert (
        quantiles_from_predict.columns.get_level_values(1).unique().tolist() == alpha
    )

    y_pred = forecaster.predict(fh=fh)
    y_mean = pred_dist.mean()
    assert_frame_equal(y_pred, y_mean)

    sparse_dist = forecaster.predict_proba(fh=sparse_fh)
    sparse_index = forecaster._check_fh(sparse_fh).to_absolute_index(forecaster.cutoff)
    assert sparse_dist.index.equals(sparse_index)
    sparse_quantiles = forecaster.predict_quantiles(fh=sparse_fh, alpha=alpha)
    assert sparse_quantiles.index.equals(sparse_index)

    near_boundary_alpha = [0.1, 0.5, 0.999]
    near_boundary_quantiles = forecaster.predict_quantiles(
        fh=fh, alpha=near_boundary_alpha
    )
    assert (
        near_boundary_quantiles.columns.get_level_values(1).unique().tolist()
        == near_boundary_alpha
    )
