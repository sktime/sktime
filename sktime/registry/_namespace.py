"""Unique namespace registry."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["_namespace"]

from functools import cache


def _namespace(include_deps=False):
    """Return the unique namespace registry.

    Parameters
    ----------
    include_deps : bool, optional (default=False)
        Whether to include dependent namespaces.

        * If False, returns namespace of ``sktime`` only.
        * If True, includes the following dependent namespaces:

            * ``scikit-learn``

    Returns
    -------
    namespace_registry : dict
        Dictionary of the combined namespace.

        Contains pointers to classes and functions from the respective namespaces.
        The keys are names, and the values are pointers.
    """
    from sktime.registry._lookup import all_estimators
    from sktime.registry._lookup_sklearn import _all_sklearn_estimators

    # retrieve all estimators from sktime and sklearn for namespace resolution
    namespace_dict_sktime = dict(all_estimators())  # noqa: F841

    if include_deps:
        namespace_dict_sklearn = dict(_all_sklearn_estimators())  # noqa: F841
        # in case of clashes sktime takes precedence
        namespace_dict_sklearn.update(namespace_dict_sktime)
        namespace_dict = namespace_dict_sklearn
    else:
        namespace_dict = namespace_dict_sktime

    return namespace_dict


@cache
def _namespace_shorthands(raise_if_not_unique=False):
    """Return the namespace for shorthand aliases.

    Retrieves shorthands from all estimators in sktime, from the tag
    ``shorthands`` if present.

    Parameters
    ----------
    raise_if_not_unique : bool, optional (default=False)
        Whether to raise an error if there is a non-unique alias in sktime.
        The True setting is used for testing purposes.

    Returns
    -------
    namespace_aliases : dict
        Dictionary of namespace aliases.

        The keys are alias names, and the values are the corresponding full namespace names.
    """
    from sktime.registry._lookup import all_estimators

    all_sktime_estimators = all_estimators(return_names=False)

    namespace_shorthands = {}

    for est in all_sktime_estimators:
        if hasattr(est, "get_class_tag"):
            shorthands = est.get_class_tag("shorthands", None)
            if shorthands is not None:
                if raise_if_not_unique:
                    est_shs = set(shorthands.keys())
                    ns_shs = set(namespace_shorthands.keys())
                    intersections = est_shs.intersection(ns_shs)
                    if intersections:
                        raise ValueError(
                            f"Error in namespace shorthands: "
                            f"Non-unique aliases found: {intersections}, "
                            f"collision with, or in, class {est.__name__}, "
                            f"please check the shorthands tag for uniqueness. "
                        )

                namespace_shorthands.update(shorthands)

    return namespace_shorthands
