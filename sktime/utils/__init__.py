__author__ = "Markus Löning"
__all__ = ["all_estimators"]

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path


def all_estimators(type_filter=None):
    """Get a list of all estimators from sktime.

    This function crawls the module and gets all classes that inherit
    from sktime's and sklearn's base classes.


    Not included are: the base classes themselves, classes defined in test modules.

    Parameters
    ----------
    type_filter : string, list of string,  or None, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'transformer' and 'forecaster' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actuall type of the class.

    References
    ----------
    ..[1]   Modified version from scikit-learn's `all_estimators()` in sklearn.utils.__init__.py
    """

    # modified from sklearn.utils.__init__

    # lazy import to avoid circular imports from sklearn.base
    import warnings
    from sklearn.base import BaseEstimator
    from sktime.forecasting.base import _BaseForecaster
    from sktime.classifiers.base import BaseClassifier
    from sktime.regressors.base import BaseRegressor
    from sktime.transformers.base import BaseTransformer

    from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests", "setup", "contrib"}
    root = str(Path(__file__).parent.parent)  # sktime package

    # Ignore deprecation warnings triggered at import time and from walking packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for importer, modname, ispkg in pkgutil.walk_packages(path=[root], prefix="sktime."):
            mod_parts = modname.split(".")

            # filter modules
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue

            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [(name, est_cls) for name, est_cls in classes
                       if not name.startswith("_")]
            all_classes.extend(classes)

    all_classes = set(all_classes)

    # only keep classes that inherit from BaseEstimator
    base_classes = ("BaseEstimator", "BaseClassifier", "BaseRegressor", "BaseTransformer", "BaseForecaster")
    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] not in base_classes)]

    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": (BaseClassifier, ClassifierMixin),
            "regressor": (BaseRegressor, RegressorMixin),
            "transformer": (BaseTransformer, TransformerMixin),
            "forecaster": _BaseForecaster
        }

        for name, base_class in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], base_class)])
        estimators = filtered_estimators

        allowed_filters = ("classifier", "regressor", "transformer", "forecaster")

        # raise error if any filter names are left
        if type_filter:
            raise ValueError(f"Parameter type_filter must be one or a list of {allowed_filters} or "
                             f"None, but found: {repr(type_filter)}")

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
