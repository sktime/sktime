# -*- coding: utf-8 -*-
"""Tests for using ForecasterTransform."""

import pytest

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_forecastertransform():
    """Test for ForecasterTransform with a None X in a forecaster pipeline.

    Same as docstring example of ForecasterTransform.
    """
    from sktime.datasets import load_longley
    from sktime.forecasting.arima import ARIMA
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.compose import ForecastingPipeline
    from sktime.forecasting.var import VAR
    from sktime.transformations.compose import ForecasterTransform

    y, X = load_longley()
    fh = ForecastingHorizon([1, 2, 3])
    pipe = ForecastingPipeline(
        steps=[
            ("exogeneous-forecast", ForecasterTransform(VAR(), fh=fh)),
            ("forecaster", ARIMA()),
        ]
    )

    pipe.fit(y, X)

    # this works without X from the future of y
    pipe.predict(fh=fh)
