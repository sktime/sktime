#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Taiwo Owoseni"]
__all__ = ["test_tuple_datatype_required_params"]

import pytest
from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.utils._testing.estimator_checks import check_required_params

pytest.mark.parametrize(
    "Estimator",
    [
        EnsembleForecaster,
        TransformedTargetForecaster,
    ],
)


def test_tuple_datatype_required_params(Estimator):
    # Check common meta-estimator interface
    with pytest.raises(TypeError):
        check_required_params(Estimator)
