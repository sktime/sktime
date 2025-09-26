"""Base class template for collections."""

__author__ = ["jgyasu"]

__all__ = ["BaseCollection"]

from abc import abstractmethod

from skbase.base import BaseObject

from sktime.classification.base import BaseClassifier
from sktime.datasets.base import BaseDataset
from sktime.forecasting.base import BaseForecaster
from sktime.performance_metrics.base import BaseMetric
from sktime.registry import craft


class BaseCollection(BaseObject):
    """Base class for collections of sktime objects."""

    _tags = {
        "authors": ["sktime developers"],
        "maintainers": ["sktime developers"],
        "object_type": "collection",
        "collection_type": None,  # "dataset loader", "estimator", "metric", or "mixed"
        "info:name": "",
        "info:description": "",
        "info:source": "",  # paper, competition, etc. (link perhaps?)
    }

    def __init__(self):
        super().__init__()
        self._cached_items = None

    @abstractmethod
    def _get(self):
        """Get the default items for this collection. Implemented by subclasses.

        Returns
        -------
        list[str]
            list of item names/ids
        """
        pass

    def get(self, item_type="all"):
        """Get items from the collection based on type.

        Parameters
        ----------
        item_type : str, default="all"
            Type of items to retrieve. Options:
            - "all": all item types
            - "dataset_loaders": dataset loader items
            - "estimators": estimator items
            - "metrics": metric items
            - Other custom types based on item attributes

        Returns
        -------
        dict[str, Any]
            dictionary with item names/ids as keys and object instances as values
        """
        if self._cached_items is None:
            names = self._get()
            self._cached_items = {name: craft(name) for name in names}

        items = self._cached_items

        if item_type == "all":
            return items
        else:
            return self._filter_items_by_type(items, item_type)

    def _filter_items_by_type(self, items, item_type):
        """Filter items by their type.

        Parameters
        ----------
        items : dict[str, Any]
            Items to filter
        item_type : str
            Type to filter by (e.g., "dataset_loaders", "estimators", "metrics")

        Returns
        -------
        dict[str, Any]
            Filtered items dictionary
        """
        filtered = {}

        for name, obj in items.items():
            if item_type == "dataset_loaders" and isinstance(obj, BaseDataset):
                filtered[name] = obj
            elif item_type == "estimators" and (
                isinstance(obj, BaseForecaster) or isinstance(obj, BaseClassifier)
            ):
                filtered[name] = obj
            elif item_type == "metrics" and isinstance(obj, BaseMetric):
                filtered[name] = obj

        return filtered

    def __len__(self):
        """Return the total number of items in the collection."""
        return len(self.get("all"))

    def __contains__(self, name):
        """Check if an item name exists in the collection."""
        all_items = self.get("all")
        return name in all_items
