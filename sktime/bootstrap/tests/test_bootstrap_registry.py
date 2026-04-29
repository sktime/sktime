"""Tests for bootstrap scitype registry.

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["sunkireddy-Barath"]

from sktime.bootstrap.base import BaseBootstrap
from sktime.registry._base_classes import (
    get_base_class_for_str,
    get_obj_scitype_list,
)


def test_bootstrap_scitype_registered():
    """Test that bootstrap is registered as a scitype."""
    assert "bootstrap" in get_obj_scitype_list()


def test_bootstrap_base_class_registered():
    """Test that the base class for bootstrap can be resolved."""
    cls = get_base_class_for_str("bootstrap")
    assert cls is BaseBootstrap
