# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Quick crafting methods to build an object from string and registry.

craft(spec)
    craft an object or estimator from string

deps(spec)
    retrieves all dependencies required to craft str, in PEP440 format

The ``craft`` function is a pair to ``str`` coercion, the two can be seen as
deserialization/serialization counterparts to each other.

That is,
spec = str(my_est)
new_est = craft(spec)

will have the same effect as new_est = spec.clone()
"""

__author__ = ["fkiraly"]

from sktime.registry._alias_str import ALIAS_DICT
from sktime.registry._craft import craft


def resolve_alias(alias):
    """Resolve an alias string to an sktime object.

    Does the following in order:

    1. check if ``alias`` is a valid key in ``registry.ALIAS_DICT``.
       If yes, returns the aliased object from ``ALIAS_DICT``, terminate.
    2. try to resolve ``alias`` via ``registry.craft``.
       If successful, return ``craft(alias)``, terminate.
    3. Raise an exception that ``alias`` is invalid.

    Parameters
    ----------
    alias : str
        a valid alias string, either:

        * a valid input to registry.craft
        * an alias string present in registry.ALIAS_DICT

    Returns
    -------
    aliased object
    """
    if not isinstance(alias, str):
        raise TypeError(
            "Error in resolve_alias, argument alias must be a str, but "
            f"found argument alias of type {type(alias)}."
        )

    # 1. check if ``alias`` is a valid key in ``registry.ALIAS_DICT``.
    # If yes, returns the aliased object from ``ALIAS_DICT``, terminate.
    if alias in ALIAS_DICT.keys():
        alias_init = ALIAS_DICT[alias][1]
        if isinstance(alias_init, str):
            return craft(alias_init)
        else:
            return ALIAS_DICT[alias][1]()

    try:
        return craft(alias)
    except Exception as e:
        raise ValueError(
            "Error in resolve_alias, argument alias must be a resolvable "
            "alias string. That is, a key of ALIAS_DICT, or resolvable by "
            f"registry.craft. Found {alias}, which is not a key of ALIAS_DICT, "
            f"and raises the following exception in registry.craft: {e}"
        )
