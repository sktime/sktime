"""Tests for FittedParamExtractor."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = []

import pytest

from sktime.datasets import load_gunpoint
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.summarize import FittedParamExtractor

X_train, y_train = load_gunpoint("train", return_X_y=True)


@pytest.mark.skipif(
    not run_test_for_class([ExponentialSmoothing, FittedParamExtractor]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("param_names", ["initial_level"])
def test_FittedParamExtractor(param_names):
    forecaster = ExponentialSmoothing()
    t = FittedParamExtractor(forecaster=forecaster, param_names=param_names)
    Xt = t.fit_transform(X_train)
    assert Xt.shape == (X_train.shape[0], len(t._check_param_names(param_names)))

    # check specific value
    forecaster.fit(X_train.iloc[47, 0])
    fitted_param = forecaster.get_fitted_params()[param_names]
    assert Xt.iloc[47, 0] == fitted_param
