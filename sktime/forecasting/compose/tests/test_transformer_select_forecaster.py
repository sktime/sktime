# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Tests for the TransformSelectForecaster"""

import pytest

from sktime.forecasting.compose import TransformSelectForecaster
from sktime.forecasting.croston import Croston
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.adi_cv import ADICVTransformer
from sktime.transformations.series.tests.test_adi_cv import (
    _generate_erratic_series,
    _generate_intermittent_series,
    _generate_smooth_series,
)


@pytest.mark.skipif(
    not run_test_for_class(TransformSelectForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters, series_generator, horizon, chosen_forecaster",
    [
        (
            {
                "smooth": NaiveForecaster(),
                "intermittent": Croston(),
            },
            _generate_smooth_series,
            50,
            NaiveForecaster(),
        ),
        (
            {
                "erratic": PolynomialTrendForecaster(),
                "lumpy": NaiveForecaster(),
            },
            _generate_erratic_series,
            50,
            PolynomialTrendForecaster(),
        ),
    ],
)
def test_forecaster_selection(
    forecasters, series_generator, horizon, chosen_forecaster
):
    """Tests if the forecaster selects the correct forecasters on the basis
    of the output from the transformer.

    Parameters
    ----------
    forecasters : dict[sktime forecasters]
        The forecasters provided by the user

    series_generator : function
        A function that generates test data

    horizon : int
        The forecasting horizon at which to predict values.

    chosen_forecaster : sktime foreacsater
       The forecaster that should be selected by the forecaster
    """
    # Defining the forecaster
    forecaster = TransformSelectForecaster(forecasters=forecasters)
    forecaster.fit(series_generator(), fh=horizon)

    # Check if the type of the chosen forecaster matches the provided forecaster type
    assert type(forecaster.chosen_forecaster_) is type(chosen_forecaster)


@pytest.mark.skipif(
    not run_test_for_class(TransformSelectForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters, series_generator, horizon, fallback_forecaster",
    [
        (
            {
                "smooth": NaiveForecaster(),
                "intermittent": Croston(),
            },
            _generate_erratic_series,
            50,
            PolynomialTrendForecaster(),
        ),
        (
            {
                "erratic": PolynomialTrendForecaster(),
                "lumpy": NaiveForecaster(),
            },
            _generate_smooth_series,
            50,
            None,
        ),
    ],
)
def test_fallback_forecaster(
    forecasters, series_generator, horizon, fallback_forecaster
):
    """This function tests the validity of the fallback forecaster and what happens
    if it is not present!

    Parameters
    ----------
    forecasters : dict[sktime forecasters]
        A dictionary consisting of sktime forecasters provided by the user

    series_generator : function
        A function that generates a time series of a particular type

    horizon : int
        The forecasting horizon at which to predict values.

    fallback_forecaster : sktime foreacsater
        A forecaster to choose if no options from `forecasters` is viable
    """
    # Creating our forecaster
    forecaster = TransformSelectForecaster(
        forecasters=forecasters, fallback_forecaster=fallback_forecaster
    )

    # Generating the required time series
    y = series_generator()

    if fallback_forecaster is None:
        with pytest.raises(ValueError):
            forecaster.fit(y, fh=horizon)

    else:
        forecaster.fit(y, fh=horizon)
        assert type(forecaster.chosen_forecaster_) is type(fallback_forecaster)


@pytest.mark.skipif(
    not run_test_for_class(TransformSelectForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters, fallback_forecaster, transformer, horizon, series_generator, params",
    [
        (
            {
                "smooth": NaiveForecaster(),
                "intermittent": Croston(),
                "erratic": NaiveForecaster(),
                "lumpy": PolynomialTrendForecaster(),
            },
            PolynomialTrendForecaster(),
            ADICVTransformer(features=["class"]),
            50,
            _generate_intermittent_series,
            {
                "smooth": NaiveForecaster,
                "intermittent": Croston,
                "erratic": NaiveForecaster,
                "lumpy": PolynomialTrendForecaster,
            },
        ),
        (
            {
                "erratic": PolynomialTrendForecaster(),
                "lumpy": NaiveForecaster(),
            },
            None,
            ADICVTransformer(features=["class", "adi"]),
            50,
            _generate_erratic_series,
            {
                "erratic": PolynomialTrendForecaster,
                "lumpy": NaiveForecaster,
            },
        ),
    ],
)
def test_get_params(
    forecasters,
    fallback_forecaster,
    transformer,
    horizon,
    series_generator,
    params,
):
    """Tests the values generated from the get_params() function.

    Parameters
    ----------
    forecasters : dict[sktime forecasters]
        A dictionary consisting of sktime forecasters provided by the user

    fallback_forecaster : sktime foreacsater
        A forecaster to choose if no options from `forecasters` is viable

    horizon : int
        The forecasting horizon at which to predict values.

    series_generator : function
        A function that generates a time series of a particular type
    """
    # Creating our forecaster
    forecaster = TransformSelectForecaster(
        forecasters=forecasters,
        transformer=transformer,
        fallback_forecaster=fallback_forecaster,
    )

    # Fit the forecaster and then fetch the generated parameters
    forecaster.fit(y=series_generator(), fh=horizon)
    forecaster_params = forecaster.get_params()

    # Check that the forecasters map correctly in the params
    for category, forecaster in forecaster_params["forecasters"].items():
        assert type(forecaster) is params[category]

    # Check the same for the fallback forecaster
    if forecaster_params["fallback_forecaster"] is None:
        assert fallback_forecaster is None

    else:
        assert type(forecaster_params["fallback_forecaster"] is fallback_forecaster)

    # Check that the extrapolated values from the transformer are of type string
    for category in forecaster_params.keys():
        assert type(category) is str
