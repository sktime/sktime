# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry lookup methods - scikit-learn estimators."""

__author__ = ["fkiraly"]
# all_estimators is also based on the sklearn utility of the same name

from functools import lru_cache

import pandas as pd
from skbase.lookup import all_objects


def _all_sklearn_estimators(
    return_names=True,
    as_dataframe=False,
    suppress_import_stdout=True,
):
    """List all scikit-learn objects in sktime and sklearn.

    This function retrieves all sklearn objects inheriting from ``BaseEstimator``,
    from the following locations:

    * the ``scikit-learn`` package, as installed in the current environment
    * the ``sktime`` package, as installed in the current environment

    Not included are: the base classes themselves, classes defined in test modules.

    Parameters
    ----------
    return_names: bool, optional (default=True)

        if True, estimator class name is included in the ``all_estimators``
        return in the order: name, estimator class, optional tags, either as
        a tuple or as pandas.DataFrame columns

        if False, estimator class name is removed from the ``all_estimators`` return.

    as_dataframe: bool, optional (default=False)

        True: ``all_estimators`` will return a ``pandas.DataFrame`` with named
        columns for all of the attributes being returned.

        False: ``all_estimators`` will return a list (either a list of
        estimators or a list of tuples, see Returns)

    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_estimators will return one of the following:

        1. list of estimators, if ``return_names=False``, and ``return_tags`` is None

        2. list of tuples (optional estimator name, class, ~ptional estimator
        tags), if ``return_names=True`` or ``return_tags`` is not ``None``.

        3. ``pandas.DataFrame`` if ``as_dataframe = True``

        if list of estimators:
            entries are estimators matching the query,
            in alphabetical order of estimator name
        if list of tuples:
            list of (optional estimator name, estimator, optional estimator
            tags) matching the query, in alphabetical order of estimator name,
            where
            ``name`` is the estimator name as string, and is an
            optional return
            ``estimator`` is the actual estimator
            ``tags`` are the estimator's values for each tag in return_tags
            and is an optional return.
        if ``DataFrame``:
            column names represent the attributes contained in each column.
            "estimators" will be the name of the column of estimators, "names"
            will be the name of the column of estimator class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.
    """  # noqa: E501
    return _all_sklearn_estimators_cached(
        return_names=return_names,
        as_dataframe=as_dataframe,
        suppress_import_stdout=suppress_import_stdout,
    )


@lru_cache(maxsize=1, typed=True)
def _all_sklearn_estimators_cached(
    return_names=True,
    as_dataframe=False,
    suppress_import_stdout=True,
):
    """List all scikit-learn objects in sktime and sklearn.

    Cached version of _all_sklearn_estimators, see above for docstring.
    """
    from sklearn.base import BaseEstimator

    MODULES_TO_IGNORE_SKLEARN = ["array_api_compat", "tests", "experimental"]
    MODULES_TO_IGNORE_SKTIME = (
        "tests",
        "setup",
        "contrib",
        "benchmarking",
        "utils",
        "all",
        "plotting",
        "_split",
        "test_split",
        "registry",
        "normal",
        "_normal",
    )

    result_sklearn = all_objects(
        object_types=BaseEstimator,
        package_name="sklearn",
        modules_to_ignore=MODULES_TO_IGNORE_SKLEARN,
        as_dataframe=as_dataframe,
        return_names=return_names,
        suppress_import_stdout=suppress_import_stdout,
    )

    result_sktime = all_objects(
        object_types=BaseEstimator,
        package_name="sktime",
        modules_to_ignore=MODULES_TO_IGNORE_SKTIME,
        as_dataframe=as_dataframe,
        return_names=return_names,
        suppress_import_stdout=suppress_import_stdout,
    )

    if as_dataframe:
        result_sklearn = pd.concat([result_sklearn, result_sktime], ignore_index=True)
    else:
        result_sklearn.extend(result_sktime)

    return result_sklearn
