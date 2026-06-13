"""A dummy classification catalogue."""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.catalogues.base import BaseCatalogue


class DummyClassificationCatalogue(BaseCatalogue):
    """Dummy catalogue of datasets, classifiers, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 4,
        "n_datasets": 1,
        "n_classifiers": 1,
        "n_metrics": 1,
        "n_cv_splitters": 1,
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        datasets = [
            "ArrowHead",
        ]

        classifiers = [{"DummyClassifier": "DummyClassifier()"}]

        metrics = [accuracy_score]

        cv_splitters = [KFold(n_splits=3)]

        all_objects = {
            "dataset": datasets,
            "classifier": classifiers,
            "metric": metrics,
            "cv_splitter": cv_splitters,
        }

        return all_objects
