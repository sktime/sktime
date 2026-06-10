"""A dummy forecasting catalogue."""

from sktime.catalogues.base import BaseCatalogue


class DummyForecastingCatalogue(BaseCatalogue):
    """Dummy catalogue of datasets, forecasters, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 5,
        "n_datasets": 1,
        "n_forecasters": 1,
        "n_metrics": 2,
        "n_cv_splitters": 1,
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        datasets = [
            "Airline",
        ]

        forecasters = [{"NaiveForecaster": "NaiveForecaster(strategy='last')"}]

        metrics = ["MeanAbsoluteError()", "MeanAbsolutePercentageError()"]

        cv_splitters = [
            "ExpandingWindowSplitter(initial_window=12, step_length=6, fh=6)"
        ]

        all_objects = {
            "dataset": datasets,
            "forecaster": forecasters,
            "metric": metrics,
            "cv_splitter": cv_splitters,
        }
        return all_objects
