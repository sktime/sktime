# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Integration-style tests for DeepDynamicFactor.

These tests aim to cover behavior that unit tests often miss:
- missing values in y
- determinism with fixed seed (smoke-level, not bitwise)
- update schema consistency when X is (not) used in fit
"""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.deep_dynamic_factor import DeepDynamicFactor
from sktime.tests.test_switch import run_test_for_class

__author__ = ["minkeymouse"]


_TEST_PARAMS = {
    "encoder_size": (8, 2),
    "max_iter": 2,
    "n_mc_samples": 2,
    "window_size": 4,
    "random_state": 42,
}


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_missing_values_in_y_fit_predict():
    """Fit/predict should work if y contains NaNs (internal interpolation)."""
    _, X = load_longley()
    y = X[["GNPDEFL", "GNP"]].iloc[:12].copy()

    # introduce some missing values
    y.iloc[3, 0] = np.nan
    y.iloc[7, 1] = np.nan

    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3])

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape == (3, 2)
    assert list(y_pred.columns) == list(y.columns)
    assert np.isfinite(y_pred.values).any()


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_determinism_same_seed_smoke():
    """Same seed + same data should give very similar predictions (smoke test)."""
    _, X = load_longley()
    y = X[["GNPDEFL", "GNP"]].iloc[:12]

    f1 = DeepDynamicFactor(**_TEST_PARAMS).fit(y)
    f2 = DeepDynamicFactor(**_TEST_PARAMS).fit(y)

    y1 = f1.predict(fh=[1, 2, 3]).to_numpy()
    y2 = f2.predict(fh=[1, 2, 3]).to_numpy()

    # Not requiring bitwise equality (depends on torch), but should be close.
    assert np.allclose(y1, y2, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_update_with_X_when_fit_with_X():
    """If fit used X, update should accept X without schema errors."""
    _, X = load_longley()
    y = X[["GNPDEFL", "GNP"]]
    X_exog = X[["UNEMP", "POP"]]

    y_train = y.iloc[:12]
    X_train = X_exog.iloc[:12]
    y_new = y.iloc[12:14]
    X_new = X_exog.iloc[12:14]

    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train, X=X_train)
    forecaster.update(y_new, X=X_new, update_params=True)

    y_pred = forecaster.predict(fh=[1])
    assert y_pred.shape == (1, y_train.shape[1])

