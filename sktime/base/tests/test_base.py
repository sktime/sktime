# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Tests for BaseObject universal base class.

tests in this module:

    test_get_class_tags - tests get_class_tags inheritance logic
    test_get_class_tag  - tests get_class_tag logic, incl default value
    test_get_tags       - tests get_tags inheritance logic
    test_get_tag        - tests get_tag logic, incl default value
    test_set_tags       - tests set_tags logic and related get_tags inheritance
"""

__author__ = ["fkiraly"]

__all__ = [
    "test_get_class_tags",
    "test_get_class_tag",
    "test_get_tags",
    "test_get_tag",
    "test_set_tags",
]

import pytest

from copy import deepcopy

from sktime.base import BaseObject


# Fixture class for testing tag system
class FixtureClassParent(BaseObject):

    _tags = {"A": "1", "B": 2, "C": 1234, 3: "D"}


# Fixture class for testing tag system, child overrides tags
class FixtureClassChild(FixtureClassParent):

    _tags = {"A": 42, 3: "E"}


FIXTURE_CLASSCHILD = FixtureClassChild

FIXTURE_CLASSCHILD_TAGS = {"A": 42, "B": 2, "C": 1234, 3: "E"}

# Fixture class for testing tag system, object overrides class tags
FIXTURE_OBJECT = FixtureClassChild()
FIXTURE_OBJECT._tags_dynamic = {"A": 42424241, "B": 3}

FIXTURE_OBJECT_TAGS = {"A": 42424241, "B": 3, "C": 1234, 3: "E"}


def test_get_class_tags():
    """Tests get_class_tags class method of BaseObject for correctness.

    Raises
    ------
    AssertError if inheritance logic in get_class_tags is incorrect
    """
    child_tags = FIXTURE_CLASSCHILD.get_class_tags()

    msg = "Inheritance logic in BaseObject.get_class_tags is incorrect"

    assert child_tags == FIXTURE_CLASSCHILD_TAGS, msg


def test_get_class_tag():
    """Tests get_class_tag class method of BaseObject for correctness.

    Raises
    ------
    AssertError if inheritance logic in get_tag is incorrect
    AssertError if default override logic in get_tag is incorrect
    """
    child_tags = dict()
    child_tags_keys = FIXTURE_CLASSCHILD_TAGS.keys()

    for key in child_tags_keys:
        child_tags[key] = FIXTURE_CLASSCHILD.get_class_tag(key)

    child_tag_default = FIXTURE_CLASSCHILD.get_class_tag("foo", "bar")
    child_tag_defaultNone = FIXTURE_CLASSCHILD.get_class_tag("bar")

    msg = "Inheritance logic in BaseObject.get_class_tag is incorrect"

    for key in child_tags_keys:
        assert child_tags[key] == FIXTURE_CLASSCHILD_TAGS[key], msg

    msg = "Default override logic in BaseObject.get_class_tag is incorrect"

    assert child_tag_default == "bar", msg
    assert child_tag_defaultNone is None, msg


def test_get_tags():
    """Tests get_tags method of BaseObject for correctness.

    Raises
    ------
    AssertError if inheritance logic in get_tags is incorrect
    """
    object_tags = FIXTURE_OBJECT.get_tags()

    msg = "Inheritance logic in BaseObject.get_tags is incorrect"

    assert object_tags == FIXTURE_OBJECT_TAGS, msg


def test_get_tag():
    """Tests get_tag method of BaseObject for correctness.

    Raises
    ------
    AssertError if inheritance logic in get_tag is incorrect
    AssertError if default override logic in get_tag is incorrect
    """
    object_tags = dict()
    object_tags_keys = FIXTURE_OBJECT_TAGS.keys()

    for key in object_tags_keys:
        object_tags[key] = FIXTURE_OBJECT.get_tag(key, raise_error=False)

    object_tag_default = FIXTURE_OBJECT.get_tag("foo", "bar", raise_error=False)
    object_tag_defaultNone = FIXTURE_OBJECT.get_tag("bar", raise_error=False)

    msg = "Inheritance logic in BaseObject.get_tag is incorrect"

    for key in object_tags_keys:
        assert object_tags[key] == FIXTURE_OBJECT_TAGS[key], msg

    msg = "Default override logic in BaseObject.get_tag is incorrect"

    assert object_tag_default == "bar", msg
    assert object_tag_defaultNone is None, msg


def test_get_tag_raises():
    """Tests that get_tag method raises error for unknown tag.

    Raises
    ------
    AssertError if get_tag does not raise error for unknown tag.
    """
    with pytest.raises(ValueError, match=r"Tag with name"):
        FIXTURE_OBJECT.get_tag("bar")


FIXTURE_TAG_SET = {"A": 42424243, "E": 3}
FIXTURE_OBJECT_SET = deepcopy(FIXTURE_OBJECT).set_tags(**FIXTURE_TAG_SET)
FIXTURE_OBJECT_SET_TAGS = {"A": 42424243, "B": 3, "C": 1234, 3: "E", "E": 3}
FIXTURE_OBJECT_SET_DYN = {"A": 42424243, "B": 3, "E": 3}


def test_set_tags():
    """Tests set_tags method of BaseObject for correctness.

    Raises
    ------
    AssertError if override logic in set_tags is incorrect
    """
    msg = "Setter/override logic in BaseObject.set_tags is incorrect"

    assert FIXTURE_OBJECT_SET._tags_dynamic == FIXTURE_OBJECT_SET_DYN, msg
    assert FIXTURE_OBJECT_SET.get_tags() == FIXTURE_OBJECT_SET_TAGS, msg
