"""A dummy classification catalogue."""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.catalogues.base import BaseCatalogue


class DummyClassificationCatalogue(BaseCatalogue):
    """Dummy catalogue of datasets, classifiers, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "catalogue_type": "mixed",
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        return all_objects


datasets = [
    "Beef",
    "ArrowHead",
]

classifiers = [
    "DummyClassifier()",
]

metrics = [accuracy_score()]

cv_splitters = [KFold(n_splits=3)]

all_objects = {
    "dataset": [f"UCRUEADataset('{dataset}')" for dataset in datasets],
    "classifier": classifiers,
    "metric": metrics,
    "cv_splitter": cv_splitters,
}
