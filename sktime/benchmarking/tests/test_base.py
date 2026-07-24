"""Tests for ``sktime.benchmarking.base``.

The module historically had a copy-paste regression: ``BaseResults.save``
constructed a ``NotImplementedError`` instance but did not raise it (see
upstream issue sktime/sktime#10169). The test below pins every
unimplemented method on ``BaseResults`` so that variant bug never returns.
"""

__author__ = ["jbbqqf"]

import pytest

from sktime.benchmarking.base import BaseResults


# Methods that are intentionally unimplemented on the base class. Each
# subclass (``HDDBaseResults`` / ``RAMResults``) overrides what it needs;
# the base must surface the gap loudly via ``NotImplementedError``.
_UNIMPLEMENTED_METHODS_AND_ARGS = [
    ("save", ()),
    (
        "save_predictions",
        ("strat", "ds", None, None, None, None, 0, "train"),
    ),
    ("load_predictions", (0, "train")),
    ("check_predictions_exist", ("strat", "ds", 0, "train")),
    ("save_fitted_strategy", ("strat", "ds", 0)),
    ("load_fitted_strategy", ("strat", "ds", 0)),
    ("check_fitted_strategy_exists", ("strat", "ds", 0)),
    ("_generate_key", ("strat", "ds", 0, "train")),
]


@pytest.mark.parametrize("method_name,args", _UNIMPLEMENTED_METHODS_AND_ARGS)
def test_base_results_unimplemented_methods_raise(method_name, args):
    """Each unimplemented BaseResults method must raise NotImplementedError."""
    instance = BaseResults()
    method = getattr(instance, method_name)
    with pytest.raises(NotImplementedError):
        method(*args)
