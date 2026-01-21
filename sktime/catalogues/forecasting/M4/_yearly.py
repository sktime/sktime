from sktime.catalogues.forecasting.M4._base import _BaseM4CompetitionCatalogue


class M4CompetitionCatalogueYearly(_BaseM4CompetitionCatalogue):
    """M4 forecasting competition catalogue for yearly time series.

    This catalogue binds the M4 yearly dataset with the standard set of
    classical forecasters and evaluates them using OWA with sp=1.
    """

    _tags = {
        "n_items": 13,
        "n_datasets": 1,
        "n_metrics": 1,
    }

    _dataset_name = "m4_yearly_dataset"
    _metric_name = "OverallWeightedAverage(sp=1)"
