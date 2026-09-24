from sktime.registry._base_classes import (
    get_base_class_for_str,
    get_obj_scitype_list,
)
from sktime.libs.tsbootstrap.base import BaseBootstrap


def test_bootstrap_scitype_registered():
    assert "bootstrap" in get_obj_scitype_list()


def test_bootstrap_base_class_registered():
    cls = get_base_class_for_str("bootstrap")

    assert cls is BaseBootstrap
