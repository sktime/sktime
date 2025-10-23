"""A dummy forecasting catalogue."""

from sktime.catalogues.base import BaseCatalogue


class DummyForecastingCatalogue(BaseCatalogue):
    """Dummy catalogue of datasets, forecasters, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "forecasting_catalogue",
        "catalogue_type": "mixed",
        "n_items": 6,
        "n_datasets": 2,
        "n_forecasters": 1,
        "n_metrics": 2,
        "n_cv_splitters": 1,
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        return all_objects


datasets = [
    "cif_2016_dataset",
    "hospital_dataset",
]

forecasters = [
    "NaiveForecaster()",
]

metrics = ["MeanAbsoluteError()", "MeanAbsolutePercentageError()"]

cv_splitters = ["ExpandingWindowSplitter(initial_window=12, step_length=6, fh=6)"]

all_objects = {
    "dataset": [f"ForecastingData('{dataset}')" for dataset in datasets],
    "forecaster": forecasters,
    "metric": metrics,
    "cv_splitter": cv_splitters,
}
