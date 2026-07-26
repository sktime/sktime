"""Composition involving parameter estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from sktime.param_est.compose._func_fitter import FunctionParamFitter
from sktime.param_est.compose._pipeline import ParamFitterPipeline

__author__ = ["fkiraly", "tpvasconcelos"]
__all__ = ["ParamFitterPipeline", "FunctionParamFitter"]
