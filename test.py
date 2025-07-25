"""Test suite to validate all the estimators."""

from sktime.registry import all_estimators

all_ests = all_estimators()
[
    x[0]
    for x in all_ests
    if (len(x[1].get_test_params()) < 2 or isinstance(x[1].get_test_params(), dict))
    and len(x[1].get_param_names()) > 0
]
