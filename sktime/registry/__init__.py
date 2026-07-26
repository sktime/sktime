"""Registry for sktime estimator base classes, tags, global aliases."""

from sktime.registry._alias import resolve_alias
from sktime.registry._alias_str import ALIAS_DICT
from sktime.registry._base_classes import (
    get_base_class_list,
    get_base_class_lookup,
    get_base_class_register,
    get_obj_scitype_list,
)
from sktime.registry._craft import craft, deps, imports
from sktime.registry._lookup import all_estimators, all_tags
from sktime.registry._scitype import is_scitype, scitype
from sktime.registry._scitype_coercion import coerce_scitype
from sktime.registry._tags import (
    ESTIMATOR_TAG_LIST,
    ESTIMATOR_TAG_REGISTER,
    check_tag_is_valid,
)

__all__ = [
    "all_estimators",
    "all_tags",
    "check_tag_is_valid",
    "coerce_scitype",
    "craft",
    "deps",
    "imports",
    "is_scitype",
    "resolve_alias",
    "scitype",
    "ALIAS_DICT",
    "ESTIMATOR_TAG_LIST",
    "ESTIMATOR_TAG_REGISTER",
    "BASE_CLASS_REGISTER",
    "BASE_CLASS_LIST",
    "BASE_CLASS_LOOKUP",
    "BASE_CLASS_SCITYPE_LIST",
    "TRANSFORMER_MIXIN_REGISTER",
    "TRANSFORMER_MIXIN_LIST",
    "TRANSFORMER_MIXIN_LOOKUP",
    "TRANSFORMER_MIXIN_SCITYPE_LIST",
]


def __getattr__(name):
    getter_dict = {
        "BASE_CLASS_LOOKUP": get_base_class_lookup,
        "BASE_CLASS_REGISTER": get_base_class_register,
        "BASE_CLASS_LIST": get_base_class_list,
        "BASE_CLASS_SCITYPE_LIST": get_obj_scitype_list,
    }
    if name in getter_dict:
        return getter_dict[name]()

    # legacy transformer mixins,
    # handled for downward compatibility
    legacy_trafo_mixin_dict = {
        "TRANSFORMER_MIXIN_LOOKUP": get_base_class_lookup,
        "TRANSFORMER_MIXIN_REGISTER": get_base_class_register,
        "TRANSFORMER_MIXIN_LIST": get_base_class_list,
        "TRANSFORMER_MIXIN_SCITYPE_LIST": get_obj_scitype_list,
    }
    if name in legacy_trafo_mixin_dict:
        return legacy_trafo_mixin_dict[name](mixin=True)

    raise AttributeError(f"module {__name__} has no attribute {name}")
