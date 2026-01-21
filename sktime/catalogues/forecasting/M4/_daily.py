from sktime.catalogues.forecasting.M4._base import _BaseM4CompetitionCatalogue


class M4CompetitionCatalogueDaily(_BaseM4CompetitionCatalogue):
    """M4 forecasting competition catalogue for daily time series.

    This catalogue binds the M4 daily dataset with the standard set of
    classical forecasters and evaluates them using OWA with sp=7.
    """

    _tags = {
        "n_items": 13,
        "n_datasets": 1,
        "n_metrics": 1,
    }

    _dataset_name = "m4_daily_dataset"
    _metric_name = "OverallWeightedAverage(sp=7)"
