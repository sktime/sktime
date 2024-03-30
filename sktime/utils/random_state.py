"""Utilities for handling the random_state variable."""
# copied from scikit-learn to avoid dependency on sklearn private methods

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies


# todo 0.29.0 - check if this can be completely replaced by skbase set_random_state
# this is feasible if skbase lower bound becomes 0.7.2 or larger
# CAUTION: default random_state current is 0, in skbase this is random_state=None
def set_random_state(estimator, random_state=0):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get_params, set_params
        Estimator with potential randomness managed by random_state parameters.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    estimator : estimator
        reference to ``estimator`` with state changed, random seed set
    """
    if _check_soft_dependencies(
        "scikit-base>=0.7.2",
        package_import_alias={"scikit-base": "skbase"},
        severity="none",
    ):
        from skbase.utils.random_state import set_random_state as _set_random_state

        return _set_random_state(estimator, random_state)

    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)

    return estimator
