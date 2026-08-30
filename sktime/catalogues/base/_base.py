"""Base class template for catalogues."""

__author__ = ["jgyasu"]

__all__ = ["BaseCatalogue"]

from abc import abstractmethod

from skbase.utils.dependencies import _check_estimator_deps

from sktime.base import BaseObject
from sktime.registry import craft


class BaseCatalogue(BaseObject):
    """Base class for catalogue objects.

    A catalogue is a curated collection of objects grouped by category,
    such as datasets, estimators, metrics, or cross-validation splitters.

    Subclasses must implement the private ``_get`` method, which returns
    the catalogue contents as a dictionary mapping category names to lists
    of entries.

    Notes
    -----
    Catalogue contents are cached after the first access. Implementations
    of ``_get`` should therefore return deterministic contents and should
    not rely on repeated invocation.
    """

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
        """Initialize the catalogue."""
        super().__init__()

        self._cached_catalogue = None
        self._cached_objects = {}

        if _check_estimator_deps(self, severity="warning"):
            self.__post_init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        pass

    @abstractmethod
    def _get(self) -> dict[str, list]:
        """Return catalogue contents grouped by category.

        Returns
        -------
        dict[str, list]
            Dictionary mapping category names to lists of catalogue entries.

            Keys are category names defined by the catalogue (for example,
            ``"forecaster"``, ``"classifier"``, ``"dataset"``, ``"metric"``,
            ``"cv_splitter"``, etc.). Values are lists containing the entries
            belonging to that category.

            Each entry may be one of the following:

            * ``str`` - a specification resolvable via ``craft``.
            * ``dict[str, Any]`` - mapping of human-given names to object
            specifications or objects.
            * object - an already-instantiated object or callable.

            For example, a forecasting catalogue may return:

            {
                "dataset": ["Airline"],
                "forecaster": [
                    {"NaiveForecaster":
                        "NaiveForecaster(strategy='last')"}
                ],
                "metric": [
                    "MeanAbsoluteError()",
                    "MeanAbsolutePercentageError()",
                ],
                "cv_splitter": [
                    "ExpandingWindowSplitter("
                    "initial_window=12, step_length=6, fh=6)"
                ],
            }

        Notes
        -----
        The returned catalogue is cached after the first access. Implementations
        should therefore return deterministic contents and must not rely on
        repeated invocation.
        """

    def get(self, object_type="all", as_object=False):
        """Retrieve entries from the catalogue.

        Parameters
        ----------
        object_type : str, default="all"
            Category of entries to retrieve.

            Valid values are:

            * ``"all"`` to retrieve entries from every category.
            * Any category returned by ``available_categories``.

        as_object : bool, default=False
            Whether to resolve catalogue specifications into objects.

            If ``False``, display names or specification strings or
            dict of {human_given_name: specification_string} are returned.

            If ``True``, catalogue entries are resolved using ``craft``
            and returned as instantiated objects or dict of {human_given_name: object}.

        Returns
        -------
        list
            Catalogue entries matching the requested category.

            If ``as_object=False``, entries are returned as display names.
            Specification strings remain unchanged, objects are converted to
            display names, and dictionary entries are returned as mappings of
            human-given names to specification strings.

            If ``as_object=True``, entries are returned as resolved
            objects. String specifications are instantiated via ``craft``,
            dictionary entries are returned as mappings of human-given names to
            resolved objects, and object entries are returned unchanged.
        """
        catalogue = self._get_catalogue()

        self._validate_object_type(object_type, catalogue)

        items = self._get_items(catalogue, object_type)

        if not as_object:
            return [self._to_name(item) for item in items]

        return self._get_objects(object_type, items)

    def available_categories(self):
        """Return the available catalogue categories.

        Returns
        -------
        list[str]
            Names of all categories defined by the catalogue. Each name can be
            passed to ``get`` via the ``object_type`` parameter to retrieve the
            corresponding catalogue entries.
        """
        return list(self._get_catalogue().keys())

    def __len__(self):
        """Return the total number of catalogue entries.

        Returns
        -------
        int
            Total number of entries across all categories.
        """
        return sum(len(items) for items in self._get_catalogue().values())

    def __contains__(self, name):
        """Check whether an entry exists in the catalogue.

        Parameters
        ----------
        name : str
            Entry name to search for.

        Returns
        -------
        bool
            True if the entry exists, otherwise False.
        """
        for item in self.get("all"):
            if isinstance(item, dict):
                if name in item:
                    return True
            elif item == name:
                return True

        return False

    def _get_catalogue(self):
        """Return cached catalogue contents.

        Returns
        -------
        dict[str, list]
            Catalogue contents grouped by category. Dictionary keys are category
            names (for example, ``"forecaster"``, ``"dataset"``, etc.), and values
            are lists of catalogue entries belonging to that category. The returned
            catalogue has been validated and cached.
        """
        if self._cached_catalogue is None:
            catalogue = self._get()

            self._validate_catalogue(catalogue)
            self._validate_tag_counts(catalogue)

            self._cached_catalogue = catalogue

        return self._cached_catalogue

    def _validate_catalogue(self, catalogue):
        """Validate the structure returned by ``_get``.

        Parameters
        ----------
        catalogue : dict
            Catalogue contents returned by ``_get``.

        Raises
        ------
        TypeError
            If the catalogue does not conform to the required structure.
        """
        if not isinstance(catalogue, dict):
            raise TypeError(
                f"{type(self).__name__}._get must return a dict, "
                f"but found {type(catalogue).__name__}."
            )

        for category, items in catalogue.items():
            if not isinstance(items, list):
                raise TypeError(
                    f"Catalogue category '{category}' must contain a list, "
                    f"but found {type(items).__name__}."
                )

    def _validate_tag_counts(self, catalogue):
        """Validate tag metadata against catalogue contents.

        Parameters
        ----------
        catalogue : dict[str, list]
            Catalogue contents.

        Raises
        ------
        ValueError
            If the ``n_items`` tag does not match the actual number of
            catalogue entries.
        """
        expected = self.get_tag("n_items")

        if expected is None:
            return

        actual = sum(len(items) for items in catalogue.values())

        if actual != expected:
            raise ValueError(
                f"Tag 'n_items' specifies {expected} items, "
                f"but catalogue contains {actual}."
            )

    def _validate_object_type(self, object_type, catalogue):
        """Validate a requested category.

        Parameters
        ----------
        object_type : str
            Requested category.
        catalogue : dict[str, list]
            Catalogue contents.

        Raises
        ------
        KeyError
            If the requested category does not exist.
        """
        if object_type == "all":
            return

        if object_type not in catalogue:
            raise KeyError(
                f"Invalid object_type '{object_type}'. "
                f"Available categories are: {list(catalogue.keys())}."
            )

    def _get_items(self, catalogue, object_type):
        """Extract entries for a category.

        Parameters
        ----------
        catalogue : dict[str, list]
            Catalogue contents.
        object_type : str
            Requested category.

        Returns
        -------
        list
            Catalogue entries belonging to the requested category. If
            ``object_type="all"``, entries from all categories are returned in a
            single flattened list.
        """
        if object_type == "all":
            return [
                item for category_items in catalogue.values() for item in category_items
            ]

        return catalogue[object_type]

    def _to_name(self, item):
        """Convert a catalogue entry to a display name.

        Parameters
        ----------
        item : Any
            Catalogue entry.

        Returns
        -------
        str or dict
            Human-readable representation of the entry.
        """
        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            return {key: self._object_name(value) for key, value in item.items()}

        return self._object_name(item)

    def _object_name(self, obj):
        """Return a readable name for an object.

        Parameters
        ----------
        obj : Any
            Object to convert.

        Returns
        -------
        str
            Readable object name.
        """
        if isinstance(obj, str):
            return obj

        if callable(obj):
            return obj.__name__

        return type(obj).__name__

    def _get_objects(self, object_type, items):
        """Resolve and cache catalogue entries.

        Parameters
        ----------
        object_type : str
            Requested category.
        items : list
            Catalogue entries.

        Returns
        -------
        list
            Catalogue entries resolved to concrete objects.

            String specifications are instantiated using ``craft``. Dictionary
            entries are returned as mappings whose keys are human-given names and
            whose values are resolved objects. Entries that are already objects
            are returned unchanged. Results are cached per ``object_type``.
        """
        if object_type not in self._cached_objects:
            self._cached_objects[object_type] = [
                self._resolve_item(item) for item in items
            ]

        return self._cached_objects[object_type]

    def _resolve_item(self, item):
        """Resolve a catalogue entry to an object.

        Parameters
        ----------
        item : Any
            Catalogue entry.

        Returns
        -------
        Any
            Resolved representation of the catalogue entry.

            String entries are instantiated using ``craft``.
            Dictionary entries are returned as mappings from human-given names to
            resolved objects.
            Existing objects are returned unchanged.
        """
        if isinstance(item, str):
            return craft(item)

        if isinstance(item, dict):
            return {
                key: craft(value) if isinstance(value, str) else value
                for key, value in item.items()
            }

        return item
