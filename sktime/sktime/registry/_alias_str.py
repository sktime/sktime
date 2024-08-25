# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dictionary of string aliases, used in _alias.resolve_alias.

New aliases should be added to ALIAS_DICT in the following format:

key is the aliased string, e.g., "my_alias"
values should be pairs (scitype, object), where object is used to construct the
    aliased object. The object can be one of:

    * a str, resolvable by ``registry.craft``

    * a callable, which constructs the aliased object upon call.
      The callable should have no arguments.
      Any necessary imports should be lazy, and inside the callable.

There should not be any imports at the top of this module.
"""

ALIAS_DICT = {}
