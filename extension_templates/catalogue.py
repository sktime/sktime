"""Extension template for catalogues.

Purpose of this implementation template:
    quick implementation of new catalogues following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new catalogue:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: any BaseObject internals
- you can add more private methods, but do not override BaseObject's private methods
  an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to sktime via PR

Mandatory method to implement:
    - _get(self): return catalogue mapping of categories -> list of items

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your catalogue

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from sktime.catalogues.base import BaseCatalogue

# todo: add any necessary imports here


# todo: change class name and write docstring
class MyCatalogue(BaseCatalogue):
    """Custom catalogue template.

    This template demonstrates a minimal concrete catalogue implementation.

    Implement `_get` to return a mapping of category name -> list of
    specification strings or objects. Specification strings are resolved via
    `sktime.registry.craft` when `get(..., as_object=True)` is called.

    Example return value from `_get`:
        {
            "estimator": ["RocketClassifier"],
            "dataset": ["ArrowHead"]
        }

    Notes
    -----
    - Items may be strings (specifiers) or actual callables/objects.
    """

    # override tags with catalogue-specific metadata
    _tags = {
        "authors": ["author1"],
        "maintainers": ["maintainer1"],
        "object_type": "catalogue",
        "catalogue_type": "example",
        "n_items": None,
        "n_classifiers": None,
        "n_forecasters": None,
        "n_datasets": None,
        "n_metrics": None,
        "n_cv_splitters": None,
        "info:name": "MyCatalogue",
        "info:description": "Example catalogue template",
        "info:source": "DOI",
    }

    # implement this
    def _get(self):
        """Return the catalogue mapping of category -> list of items.

        Returns
        -------
        dict
            mapping from category name (str) to list of items. Items can be
            - strings: specification strings understood by `sktime.registry.craft`
            - callables/objects: returned as-is by `get(..., as_object=True)`

        Example
        -------
        return {
            "estimator": ["DummyClassifier"],
            "dataset": ["Airline"]
            "metrics": ["MeanAbsoluteError"]
        }
        """
        # TODO: replace the example entries below with catalogue contents
        return {
            "estimators": [
                "DummyClassifier",
            ],
            "datasets": [
                "ArrowHead",
            ],
            "metrics": [
                "MeanAbsoluteError",
            ],
        }
