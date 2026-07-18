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
        "capability:sample_weight": False,  # ability to handle sample weights in fit
        "capability:random_state": False,  # has a random_state parameter?
        "property:randomness": "deterministic",  # or "stochastic"/"derandomized"
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._paramc
        self.paramc = paramc

        # leave this as is
        super().__init__()

        # do not put anything else in __init__,
        # use __dynamic_tags__ for dynamic tag setting
        # use __post_init__ for any further initialization logic

    # todo: add if there is dynamic tag setting logic, otherwise delete this method
    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        # todo: if tags of estimator depend on component tags, set these here
        #  typically only needed if estimator is a composite
        #  tags set here apply to the instance, and override the class tags
        #
        # example 1: conditional setting of a tag based on parameter foo
        # if self.foo == 42:
        #   self.set_tags(**{"capability:missing_values": True})
        # example 2: cloning tags from component estimator component_estimator
        #   self.clone_tags(self.component_estimator, ["capability:missing_values"])

    # todo: add any post-init logic here, otherwise delete this method
    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        # todo: optional, parameter checking or coercion should happen here
        # if writes derived values to self, should *not* overwrite self.paramc etc
        # instead, write to self._paramc, self._newparam (starting with _)
        # example of handling conditional parameters or mutable defaults:
        if self.paramc is None:
            from sktime.somewhere import MyOtherEstimator

            self._paramc = MyOtherEstimator(foo=42)
        else:
            # estimators should be cloned to avoid side effects
            self._paramc = self.paramc.clone()

    # TODO: implement this
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
