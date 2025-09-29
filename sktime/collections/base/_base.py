"""Base class template for collections."""

__author__ = ["jgyasu"]

__all__ = ["BaseCollection"]

from abc import abstractmethod

from skbase.base import BaseObject

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

    def get(self, item_type="all", as_object=False):
        """Get items from the collection based on type.

        Parameters
        ----------
        item_type : str, default="all"
            Type of items to retrieve. Options:
            - "all": all item types
            - or one of categories returned by `available_categories` method

        as_object : bool, default=False
            If True, return the crafted object instances.
            Otherwise, return item names as strings.

        Returns
        -------
        list[str] or list[Any]
            List of names (default) or crafted objects.
        """
        names_dict = self._get()

        if item_type != "all" and item_type not in names_dict:
            raise KeyError(
                f"Invalid item_type '{item_type}'. "
                f"Available keys: {list(names_dict.keys())}"
            )

        if not as_object:
            if item_type == "all":
                return [name for names in names_dict.values() for name in names]
            else:
                return names_dict[item_type]

        if self._cached_items is None:
            self._cached_items = {}

        if item_type == "all":
            all_objects = []
            for key, names in names_dict.items():
                if key not in self._cached_items:
                    self._cached_items[key] = [craft(name) for name in names]
                all_objects.extend(self._cached_items[key])
            return all_objects
        else:
            if item_type not in self._cached_items:
                self._cached_items[item_type] = [
                    craft(name) for name in names_dict[item_type]
                ]
            return self._cached_items[item_type]

    def available_categories(self):
        """Return the available item categories in the collection.

        Returns
        -------
        list[str]
            List of keys from _get(), e.g., ['datasets', 'estimators', 'metrics'].
        """
        return list(self._get().keys())

    def __len__(self):
        """Return the total number of items in the collection."""
        return len(self.get("all"))

    def __contains__(self, name):
        """Check if an item name exists in the collection."""
        all_items = self.get("all")
        return name in all_items
