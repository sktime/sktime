"""TSC Bakeoff collection."""

from sktime.collections.base import BaseCollection


class TSCCBakeoff(BaseCollection):
    """Collection of datasets, estimators, and metrics from TSC Bakeoff."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "collection_type": "mixed",
        "info:name": "TSC Bakeoff Collection",
        "info:source": "http://www.timeseriesclassification.com/",
    }

    def _get(self):
        """Return a list of names (datasets, estimators, metrics)."""
        return [
            # Dataset loaders
            "UCRUEADataset('Beef')",
            "UCRUEADataset('Coffee')",
            "UCRUEADataset('ECG200')",
            "UCRUEADataset('Adiac')",
            "UCRUEADataset('GunPoint')",
            # Estimators (classifiers)
            "KNeighborsTimeSeriesClassifier",
            "TimeSeriesForestClassifier",
        ]
