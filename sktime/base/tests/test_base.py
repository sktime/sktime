# -*- coding: utf-8 -*-
"""
Tests for BaseObject universal base class.

tests in this module:

    test_get_class_tags - tests get_class_tags inheritance logic
    test_get_class_tag  - tests get_class_tag logic, incl default value
    test_get_tags       - tests get_tags inheritance logic
    test_get_tag        - tests get_tag logic, incl default value
    test_set_tag        - tests set_tag logic and related get_tags inheritance

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["fkiraly"]

from copy import deepcopy

from sktime.base import BaseObject
from sktime.utils._testing.deep_equals import deep_equals


# Fixture class for testing tag system
class FixtureClassParent(BaseObject):

    _tags = {"A": "1", "B": 2, "C": [], 3: "D"}


# Fixture class for testing tag system, child overrides tags
class FixtureClassChild(FixtureClassParent):

    _tags = {"A": 42, 3: "E"}


FIXTURE_CLASSCHILD = FixtureClassChild

FIXTURE_CLASSCHILD_TAGS = {"A": 42, "B": 2, "C": [], 3: "E"}

# Fixture class for testing tag system, object overrides class tags
FIXTURE_OBJECT = FixtureClassChild()
FIXTURE_OBJECT._tags_dynamic = {"A": 42424241, "B": 3}

FIXTURE_OBJECT_TAGS = {"A": 42424241, "B": 3, "C": [], 3: "E"}


def test_get_class_tags():
    """Tests get_class_tags class method of BaseObject for correctness

    Raises
    ------
    AssertError if inheritance logic in get_class_tags is incorrect
    """
    child_tags = FIXTURE_CLASSCHILD.get_class_tags()

    msg = "Inheritance logic in BaseObject.get_class_tags is incorrect"

    assert deep_equals(child_tags, FIXTURE_CLASSCHILD_TAGS), msg


def test_get_class_tag():
    """Tests get_class_tag class method of BaseObject for correctness

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
        assert deep_equals(child_tags[key], FIXTURE_CLASSCHILD_TAGS[key]), msg

    msg = "Default override logic in BaseObject.get_class_tag is incorrect"

    assert deep_equals(child_tag_default, "bar"), msg
    assert child_tag_defaultNone is None, msg


def test_get_tags():
    """Tests get_tags method of BaseObject for correctness

    Raises
    ------
    AssertError if inheritance logic in get_tags is incorrect
    """
    object_tags = FIXTURE_OBJECT.get_tags()

    msg = "Inheritance logic in BaseObject.get_tags is incorrect"

    assert deep_equals(object_tags, FIXTURE_OBJECT_TAGS), msg


def test_get_tag():
    """Tests get_tag method of BaseObject for correctness

    Raises
    ------
    AssertError if inheritance logic in get_tag is incorrect
    AssertError if default override logic in get_tag is incorrect
    """
    object_tags = dict()
    object_tags_keys = FIXTURE_OBJECT_TAGS.keys()

    for key in object_tags_keys:
        object_tags[key] = FIXTURE_OBJECT.get_tag(key)

    object_tag_default = FIXTURE_OBJECT.get_tag("foo", "bar")
    object_tag_defaultNone = FIXTURE_OBJECT.get_tag("bar")

    msg = "Inheritance logic in BaseObject.get_tag is incorrect"

    for key in object_tags_keys:
        assert deep_equals(object_tags[key], FIXTURE_OBJECT_TAGS[key]), msg

    msg = "Default override logic in BaseObject.get_tag is incorrect"

    assert deep_equals(object_tag_default, "bar"), msg
    assert object_tag_defaultNone is None, msg


FIXTURE_TAG_SET = {"A": 42424243, "E": 3}
FIXTURE_OBJECT_SET = deepcopy(FIXTURE_OBJECT).set_tag(**FIXTURE_TAG_SET)
FIXTURE_OBJECT_SET_TAGS = {"A": 42424243, "B": 3, "C": [], 3: "E", "E": 3}
FIXTURE_OBJECT_SET_DYN = {"A": 42424243, "B": 3, "E": 3}


def test_set_tag():
    """Tests get_set method of BaseObject for correctness

    Raises
    ------
    AssertError if set_tag override logic in set_tag is incorrect
    """

    msg = "Setter/override in BaseObject.set_tag is incorrect"

    assert deep_equals(FIXTURE_OBJECT_SET._tags_dynamic, FIXTURE_OBJECT_SET_DYN), msg
    assert deep_equals(FIXTURE_OBJECT_SET.get_tags(), FIXTURE_OBJECT_SET_TAGS), msg
