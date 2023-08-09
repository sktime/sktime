# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dictionary of string aliases, used in _alias.resolve_alias.

New aliases should be added to ALIAS_DICT in the following format:

key is the aliased string, e.g., "my_alias"
values should be callables that construct the alias.
    Any necessary imports should be lazy, and inside the callable.
    There should not be any imports at the top of this module.
"""

ALIAS_DICT = {}
