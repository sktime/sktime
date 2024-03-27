# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Module for nested equality checking."""
from sktime.utils.validation._dependencies import _check_soft_dependencies

# todo 0.29.0: check whether scikit-base>=0.6.1 lower bound is 0.6.1 or higher
# if yes, remove legacy handling and only use the new deep_equals
if _check_soft_dependencies(
    "scikit-base<0.6.1",
    package_import_alias={"scikit-base": "skbase"},
    severity="none",
):
    from sktime.utils._testing.deep_equals import deep_equals

else:
    from sktime.utils.deep_equals._deep_equals import deep_equals

__all__ = [
    "deep_equals",
]
