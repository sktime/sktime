#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from sktime.transformers.summarise import FittedParamExtractor
from sktime.datasets import load_gunpoint
from sktime.forecasting.exp_smoothing import ExponentialSmoothingForecaster
import pytest

X_train, y_train = load_gunpoint("TRAIN", return_X_y=True)


@pytest.mark.parametrize("param_names", ["smoothing_level"])
def test_FittedParamExtractor(param_names):
    forecaster = ExponentialSmoothingForecaster()
    t = FittedParamExtractor(forecaster=forecaster, param_names=param_names)
    Xt = t.fit_transform(X_train)

    assert X_train.shape[0] == Xt.shape[0]
    assert Xt.iloc[0, 0] == forecaster.fit(X_train.iloc[0, 0]).get_fitted_params().get(param_names)
