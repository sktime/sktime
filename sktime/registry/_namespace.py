"""Unique namespace registry."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["_namespace"]


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
