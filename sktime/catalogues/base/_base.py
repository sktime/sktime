"""Base class template for Catalogues."""

__author__ = ["jgyasu"]

__all__ = ["BaseCatalogue"]

from abc import abstractmethod

from sktime.base import BaseObject
from sktime.registry import craft


class BaseCatalogue(BaseObject):
    """Base class for catalogues."""

    _tags = {
        "authors": ["sktime developers"],
        "maintainers": ["sktime developers"],
        "object_type": "catalogue",
        "catalogue_type": None,
        "n_items": None,
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
        dict
            Dictionary of categories containing lists of item names/ids, dicts,
            or objects.
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
            If False, return specification strings or function names.

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

        items = (
            [item for items in names_dict.values() for item in items]
            if object_type == "all"
            else names_dict[object_type]
        )

        if not as_object:
            res = []
            for item in items:
                if isinstance(item, str):
                    res.append(item)
                elif isinstance(item, dict):
                    # keep the custom ID mapping but resolve the estimator repr
                    res.append(
                        {
                            k: (
                                v
                                if isinstance(v, str)
                                else (v.__name__ if callable(v) else type(v).__name__)
                            )
                            for k, v in item.items()
                        }
                    )
                else:
                    res.append(item.__name__ if callable(item) else type(item).__name__)
            return res

        # as_object=True path
        if self._cached_objects is None:
            self._cached_objects = {}

        if object_type not in self._cached_objects:
            processed = []
            for item in items:
                # handle dictionaries of estimators mapped to custom IDs
                if isinstance(item, dict):
                    processed_dict = {}
                    for est_id, est in item.items():
                        if isinstance(est, str):
                            est = craft(est)
                        processed_dict[est_id] = est
                    processed.append(processed_dict)

                # handle string specs
                elif isinstance(item, str):
                    processed.append(craft(item))

                # handle raw objects/callables
                else:
                    processed.append(item)
            self._cached_objects[object_type] = processed

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
        # check against both standard list items and dictionary keys/values
        for item in all_items:
            if item == name:
                return True
            if isinstance(item, dict) and (name in item or name in item.values()):
                return True
        return False
