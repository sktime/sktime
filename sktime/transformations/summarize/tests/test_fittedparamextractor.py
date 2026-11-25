"""Tests for FittedParamExtractor."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "shubham5080"]
__all__ = []

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.datasets import load_gunpoint
from sktime.forecasting.trend import TrendForecaster
from sktime.tests.test_switch import run_test_for_class, run_test_module_changed
from sktime.transformations.summarize import FittedParamExtractor

X_train, y_train = load_gunpoint("train", return_X_y=True)


@pytest.mark.skipif(
    not run_test_for_class([FittedParamExtractor])
    or not run_test_module_changed("sktime.forecasting.exp_smoothing")
    or not _check_estimator_deps(FittedParamExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("param_names", ["initial_level"])
def test_FittedParamExtractor(param_names):
    """Test FittedParamExtractor with ExponentialSmoothing."""
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    forecaster = ExponentialSmoothing()
    t = FittedParamExtractor(forecaster=forecaster, param_names=param_names)
    Xt = t.fit_transform(X_train)
    assert Xt.shape == (X_train.shape[0], len(t._check_param_names(param_names)))

    # check specific value
    forecaster.fit(X_train.iloc[47, 0])
    fitted_param = forecaster.get_fitted_params()[param_names]
    assert Xt.iloc[47, 0] == fitted_param


@pytest.mark.skipif(
    not run_test_for_class([FittedParamExtractor]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fittedparam_df_list_and_numpy3d():
    """Regression: X_inner_mtype is df-list/numpy3D; both paths extract params."""
    t = FittedParamExtractor(
        forecaster=TrendForecaster(), param_names="regressor__intercept"
    )
    assert t.get_tag("X_inner_mtype") == ["df-list", "numpy3D"]

    # df-list of univariate DataFrames
    df_list = [
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}),
        pd.DataFrame({"y": [10.0, 8.0, 6.0, 4.0]}),
    ]
    Xt_list = t.fit_transform(df_list)
    assert Xt_list.shape == (2, 1)
    np.testing.assert_allclose(Xt_list.iloc[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(Xt_list.iloc[1, 0], 10.0, atol=1e-6)

    # numpy3D: (n_instances, n_vars, n_timepoints)
    X3 = np.array(
        [[[1.0, 2.0, 3.0, 4.0]], [[10.0, 8.0, 6.0, 4.0]]],
        dtype=float,
    )
    t2 = FittedParamExtractor(
        forecaster=TrendForecaster(), param_names="regressor__intercept"
    )
    Xt3 = t2.fit_transform(X3)
    assert Xt3.shape == (2, 1)
    np.testing.assert_allclose(Xt3.iloc[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(Xt3.iloc[1, 0], 10.0, atol=1e-6)
