"""Tests for FittedParamExtractor."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = []

import pytest

from sktime.datasets import load_gunpoint
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.panel.summarize import FittedParamExtractor
from sktime.utils.validation._dependencies import _check_estimator_deps

X_train, y_train = load_gunpoint("train", return_X_y=True)


@pytest.mark.skipif(
    not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
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
