from sktime.catalogues.forecasting.M4._base import _BaseM4CompetitionCatalogue


class M4CompetitionCatalogueHourly(_BaseM4CompetitionCatalogue):
    """M4 forecasting competition catalogue for hourly time series.

    The M4 competition is a large-scale forecasting benchmark that evaluates
    forecast accuracy across multiple temporal granularities.

    This catalogue binds the M4 hourly dataset with the standard set of
    classical forecasters and evaluates them using OWA with sp=24.

    Examples
    --------
    >>> from sktime.catalogues.forecasting.M4 import M4CompetitionCatalogueHourly
    >>> cat = M4CompetitionCatalogueHourly()
    >>> len(cat)
    13
    >>> cat.get("dataset")
    ["ForecastingData('m4_hourly_dataset')"]
    >>> "Naive_1" in cat
    True
    """

    _tags = {
        "n_items": 13,
        "n_datasets": 1,
        "n_metrics": 3,
        "n_forecasters": 1,
    }

    _dataset_name = "m4_hourly_dataset"
    _metric_name = [
        "OverallWeightedAverage(sp=24)",
        "MeanAbsolutePercentageError(symmetric=True)",
        "MeanAbsoluteScaledError()",
    ]
    _specific_forecasters = [
        {"Naive_S": "NaiveForecaster(strategy='last', sp=24)"},
    ]
