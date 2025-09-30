"""Base class template for Catalogues."""

__author__ = ["jgyasu"]

__all__ = ["BaseCatalogue"]

from abc import abstractmethod

from skbase.base import BaseObject

from sktime.registry import craft


class BaseCatalogue(BaseObject):
    """Base class for catalogues."""

    _tags = {
        "authors": ["sktime developers"],
        "maintainers": ["sktime developers"],
        "object_type": "catalogue",
        "catalogue_type": None,
        "info:name": "",
        "info:description": "",
        "info:source": "",  # DOI
    }

    def __init__(self):
        super().__init__()
        self._cached_objects = None

    @abstractmethod
    def _get(self):
        """Get the default items for this catalogue. Implemented by subclasses.

        Returns
        -------
        list[str]
            list of item names/ids
        """
        pass

    def get(self, object_type="all", as_object=False):
        """Get objects from the catalogue based on type.

        Parameters
        ----------
        object_type : str, default="all"
            Type of objects to retrieve. Options:

            - "all": all object types
            - or one of categories returned by `available_categories` method

        as_object : bool, default=False
            If True, return object instances.
            If False, return specification strings that can be turned
            into objects using `craft`.

        Returns
        -------
        list[str] or list[Any]
            List of specification names (default) or object instances.
        """
        names_dict = self._get()

        if object_type != "all" and object_type not in names_dict:
            raise KeyError(
                f"Invalid object_type '{object_type}'. "
                f"Available keys: {list(names_dict.keys())}"
            )

        if not as_object:
            if object_type == "all":
                return [name for names in names_dict.values() for name in names]
            else:
                return names_dict[object_type]

        if self._cached_objects is None:
            self._cached_objects = {}

        if object_type == "all":
            all_objects = []
            for key, names in names_dict.items():
                if key not in self._cached_objects:
                    self._cached_objects[key] = [craft(name) for name in names]
                all_objects.extend(self._cached_objects[key])
            return all_objects
        else:
            if object_type not in self._cached_objects:
                self._cached_objects[object_type] = [
                    craft(name) for name in names_dict[object_type]
                ]
            return self._cached_objects[object_type]

    def available_categories(self):
        """Return the available item categories in the catalogue.

        Returns
        -------
        list[str]
            List of keys from _get(), e.g., ['dataset', 'estimator', 'metric'].
        """
        return list(self._get().keys())

    def __len__(self):
        """Return the total number of items in the catalogue."""
        return len(self.get("all"))

    def __contains__(self, name):
        """Check if an item name exists in the catalogue."""
        all_items = self.get("all")
        return name in all_items
