from sktime.catalogues.forecasting.M4._base import _BaseM4CompetitionCatalogue


class M4CompetitionCatalogueMonthly(_BaseM4CompetitionCatalogue):
    """M4 forecasting competition catalogue for monthly time series.

    The M4 competition is a large-scale forecasting benchmark that evaluates
    forecast accuracy across multiple temporal granularities.

    This catalogue binds the M4 monthly dataset with the standard set of
    classical forecasters and evaluates them using OWA with sp=12.
    """

    _tags = {
        "n_items": 14,
        "n_datasets": 1,
        "n_metrics": 1,
    }

    _dataset_name = "m4_monthly_dataset"
    _metric_name = "OverallWeightedAverage(sp=12)"
