# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests the aliasing logic in the Tag Aliaser."""

from sktime.base import BaseObject as _BaseObject
from sktime.base._base import TagAliaserMixin as _TagAliaserMixin


class AliaserTestClass(_TagAliaserMixin, _BaseObject):
    """Class for testing tag aliasing logic."""

    _tags = {
        "new_tag_1": "new_tag_1_value",
        "old_tag_1": "old_tag_1_value",
        "new_tag_2": "new_tag_2_value",
        "old_tag_3": "old_tag_3_value",
    }

    alias_dict = {
        "old_tag_1": "new_tag_1",
        "old_tag_2": "new_tag_2",
        "old_tag_3": "new_tag_3",
    }
    deprecate_dict = {
        "old_tag_1": "42.0.0",
        "old_tag_2": "84.0.0",
        "old_tag_3": "126.0.0",
    }

    def __init__(self):
        super().__init__()


def test_tag_aliaser():
    """Tests the tag aliaser logic, as described in its docstring."""
    # case both new and old tags exist
    # old tag takes precedence
    new_tag_1_val = AliaserTestClass().get_tag("new_tag_1")
    assert new_tag_1_val == "old_tag_1_value"
    old_tag_1_val = AliaserTestClass().get_tag("old_tag_1")
    assert old_tag_1_val == "old_tag_1_value"

    new_tag_1_val = AliaserTestClass.get_class_tag("new_tag_1")
    assert new_tag_1_val == "old_tag_1_value"
    old_tag_1_val = AliaserTestClass.get_class_tag("old_tag_1")
    assert old_tag_1_val == "old_tag_1_value"

    # case only new tag exists
    new_tag_2_val = AliaserTestClass().get_tag("new_tag_2")
    assert new_tag_2_val == "new_tag_2_value"
    old_tag_2_val = AliaserTestClass().get_tag("old_tag_2")
    assert old_tag_2_val == "new_tag_2_value"

    new_tag_2_val = AliaserTestClass.get_class_tag("new_tag_2")
    assert new_tag_2_val == "new_tag_2_value"
    old_tag_2_val = AliaserTestClass.get_class_tag("old_tag_2")
    assert old_tag_2_val == "new_tag_2_value"

    # case only old tag exists
    new_tag_3_val = AliaserTestClass().get_tag("new_tag_3")
    assert new_tag_3_val == "old_tag_3_value"
    old_tag_3_val = AliaserTestClass().get_tag("old_tag_3")
    assert old_tag_3_val == "old_tag_3_value"

    new_tag_3_val = AliaserTestClass.get_class_tag("new_tag_3")
    assert new_tag_3_val == "old_tag_3_value"
    old_tag_3_val = AliaserTestClass.get_class_tag("old_tag_3")
    assert old_tag_3_val == "old_tag_3_value"

    # test all tags retrieval
    all_tags = AliaserTestClass().get_tags()
    assert all_tags["new_tag_1"] == "old_tag_1_value"
    assert all_tags["old_tag_1"] == "old_tag_1_value"
    assert all_tags["new_tag_2"] == "new_tag_2_value"
    assert all_tags["old_tag_2"] == "new_tag_2_value"
    assert all_tags["new_tag_3"] == "old_tag_3_value"
    assert all_tags["old_tag_3"] == "old_tag_3_value"

    all_tags_cls = AliaserTestClass.get_class_tags()
    assert all_tags_cls["new_tag_1"] == "old_tag_1_value"
    assert all_tags_cls["old_tag_1"] == "old_tag_1_value"
    assert all_tags_cls["new_tag_2"] == "new_tag_2_value"
    assert all_tags_cls["old_tag_2"] == "new_tag_2_value"
    assert all_tags_cls["new_tag_3"] == "old_tag_3_value"
    assert all_tags_cls["old_tag_3"] == "old_tag_3_value"


class NoneTagTestClass(_BaseObject):
    """Class for testing tag_value_default when tag is None."""

    _tags = {
        "none_tag": None,
        "real_tag": "real_value",
    }


def test_get_class_tag_default_when_none():
    """Test that tag_value_default is returned when tag value is None.

    Regression test for #10305: get_class_tag returns None even if
    tag_value_default is passed, when the tag is explicitly set to None
    in _tags (e.g. "python_version": None).
    """
    # When tag is None and a non-None default is provided, return the default
    result = NoneTagTestClass.get_class_tag("none_tag", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # When tag is None and no default is provided, return None
    result = NoneTagTestClass.get_class_tag("none_tag")
    assert result is None, f"Expected None, got {result!r}"

    # When tag has a real value, return that value even if default is provided
    result = NoneTagTestClass.get_class_tag("real_tag", "fallback")
    assert result == "real_value", f"Expected 'real_value', got {result!r}"

    # Non-existent tag with default returns the default
    result = NoneTagTestClass.get_class_tag("nonexistent", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # Non-existent tag without default returns None
    result = NoneTagTestClass.get_class_tag("nonexistent")
    assert result is None, f"Expected None, got {result!r}"


def test_get_tag_default_when_none():
    """Test that tag_value_default is returned when tag value is None for get_tag.

    Same regression as #10305 but for the instance-level get_tag method.
    """
    obj = NoneTagTestClass()

    # When tag is None and a non-None default is provided, return the default
    result = obj.get_tag("none_tag", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # When tag is None and no default is provided, return None
    result = obj.get_tag("none_tag")
    assert result is None, f"Expected None, got {result!r}"

    # When tag has a real value, return that value even if default is provided
    result = obj.get_tag("real_tag", "fallback")
    assert result == "real_value", f"Expected 'real_value', got {result!r}"
