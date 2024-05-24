# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Tests for the CategoryCompositor"""

import pytest

from sktime.forecasting.compose import CategoryCompositor
from sktime.forecasting.croston import Croston
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.tests.test_adi_cv import (
    _generate_erratic_series,
    _generate_smooth_series,
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
    """Tests if the compositor selects the correct forecasters on the basis
    of the output from the transformer.

    Parameters
    ----------
    forecasters : dict[sktime forecasters]
        The forecasters provided by the user

    series_generator : function
        A function that generates test data
    """

    # Defining the compositor
    compositor = CategoryCompositor(forecasters=forecasters)
    compositor.fit(series_generator(), fh=horizon)

    # Check if the type of the chosen forecaster matches the provided forecaster type
    assert type(compositor.chosen_forecaster_) is type(chosen_forecaster)


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

    fallback_forecaster : sktime foreacsater
        A forecaster to choose if no options from `forecasters` is viable
    """
    # Creating our compositor
    compositor = CategoryCompositor(
        forecasters=forecasters, fallback_forecaster=fallback_forecaster
    )

    # Generating the required time series
    y = series_generator()

    if fallback_forecaster is None:
        with pytest.raises(ValueError):
            compositor.fit(y, fh=horizon)

    else:
        compositor.fit(y, fh=horizon)
        assert type(compositor.chosen_forecaster_) is type(fallback_forecaster)
