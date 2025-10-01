"""An example catalogue."""

from sktime.catalogues.base import BaseCatalogue


class ExampleCatalogue(BaseCatalogue):
    """catalogue of datasets, estimators, and metrics from TSC Bake Off 2017."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "catalogue_type": "mixed",
    }

    def _get(self):
        """Return a dict of items (datasets, estimators, metrics)."""
        return items


datasets = [
    "Beef",
    "ArrowHead",
]

classifiers = [
    "DummyClassifier",
    "KNeighborsTimeSeriesClassifier()",
]

items = {
    "dataset": [f"UCRUEADataset('{dataset}')" for dataset in datasets],
    "classifier": classifiers,
}
