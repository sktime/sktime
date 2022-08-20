# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Tests for BaseObject universal base class.

tests in this module:

    test_get_class_tags  - tests get_class_tags inheritance logic
    test_get_class_tag   - tests get_class_tag logic, incl default value
    test_get_tags        - tests get_tags inheritance logic
    test_get_tag         - tests get_tag logic, incl default value
    test_set_tags        - tests set_tags logic and related get_tags inheritance

    test_reset           - tests reset logic on a simple, non-composite estimator
    test_reset_composite - tests reset logic on a composite estimator
"""

__author__ = ["fkiraly"]

__all__ = [
    "test_get_class_tags",
    "test_get_class_tag",
    "test_get_tags",
    "test_get_tag",
    "test_set_tags",
    "test_reset",
    "test_reset_composite",
]

from copy import deepcopy

import pytest

from sktime.base import BaseEstimator, BaseObject


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
    AssertionError if override logic in set_tags is incorrect
    """
    msg = "Setter/override logic in BaseObject.set_tags is incorrect"

    assert FIXTURE_OBJECT_SET._tags_dynamic == FIXTURE_OBJECT_SET_DYN, msg
    assert FIXTURE_OBJECT_SET.get_tags() == FIXTURE_OBJECT_SET_TAGS, msg


class CompositionDummy(BaseObject):
    """Potentially composite object, for testing."""

    def __init__(self, foo, bar=84):
        self.foo = foo
        self.foo_ = deepcopy(foo)
        self.bar = bar


def test_is_composite():
    """Tests is_composite tag for correctness.

    Raises
    ------
    AssertionError if logic behind is_composite is incorrect
    """
    non_composite = CompositionDummy(foo=42)
    composite = CompositionDummy(foo=non_composite)

    assert not non_composite.is_composite()
    assert composite.is_composite()


class ResetTester(BaseObject):

    clsvar = 210

    def __init__(self, a, b=42):
        self.a = a
        self.b = b
        self.c = 84

    def foo(self, d=126):
        self.d = deepcopy(d)
        self._d = deepcopy(d)
        self.d_ = deepcopy(d)
        self.f__o__o = 252


def test_reset():
    """Tests reset method for correct behaviour, on a simple estimator.

    Raises
    ------
    AssertionError if logic behind reset is incorrect, logic tested:
        reset should remove any object attributes that are not hyper-parameters,
        with the exception of attributes containing double-underscore "__"
        reset should not remove class attributes or methods
        reset should set hyper-parameters as in pre-reset state
    """
    x = ResetTester(168)
    x.foo()

    x.reset()

    assert hasattr(x, "a") and x.a == 168
    assert hasattr(x, "b") and x.b == 42
    assert hasattr(x, "c") and x.c == 84
    assert hasattr(x, "clsvar") and x.clsvar == 210
    assert not hasattr(x, "d")
    assert not hasattr(x, "_d")
    assert not hasattr(x, "d_")
    assert hasattr(x, "f__o__o") and x.f__o__o == 252
    assert hasattr(x, "foo")


def test_reset_composite():
    """Test reset method for correct behaviour, on a composite estimator."""
    y = ResetTester(42)
    x = ResetTester(a=y)

    x.foo(y)
    x.d.foo()

    x.reset()

    assert hasattr(x, "a")
    assert not hasattr(x, "d")
    assert not hasattr(x.a, "d")


def test_components():
    """Tests component retrieval.

    Raises
    ------
    AssertionError if logic behind _components is incorrect, logic tested:
        calling _components on a non-composite returns an empty dict
        calling _components on a composite returns name/BaseObject pair in dict,
        and BaseObject returned is identical with attribute of the same name
    """
    non_composite = CompositionDummy(foo=42)
    composite = CompositionDummy(foo=non_composite)

    non_comp_comps = non_composite._components()
    comp_comps = composite._components()

    assert isinstance(non_comp_comps, dict)
    assert set(non_comp_comps.keys()) == set()

    assert isinstance(comp_comps, dict)
    assert set(comp_comps.keys()) == set(["foo_"])
    assert comp_comps["foo_"] == composite.foo_
    assert comp_comps["foo_"] != composite.foo


class FittableCompositionDummy(BaseEstimator):
    """Potentially composite object, for testing."""

    def __init__(self, foo, bar=84):
        self.foo = foo
        self.foo_ = deepcopy(foo)
        self.bar = bar

    def fit(self):
        if hasattr(self.foo_, "fit"):
            self.foo_.fit()
        self._is_fitted = True


def test_get_fitted_params():
    """Tests fitted parameter retrieval.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on a non-composite fittable returns the fitted param
        calling get_fitted_params on a composite returns all nested params
    """
    non_composite = FittableCompositionDummy(foo=42)
    composite = FittableCompositionDummy(foo=deepcopy(non_composite))

    non_composite.fit()
    composite.fit()

    non_comp_f_params = non_composite.get_fitted_params()
    comp_f_params = composite.get_fitted_params()

    assert isinstance(non_comp_f_params, dict)
    assert set(non_comp_f_params.keys()) == set(["foo"])

    assert isinstance(comp_f_params, dict)
    assert set(comp_f_params) == set(["foo", "foo__foo"])
    assert comp_f_params["foo"] == composite.foo_
    assert comp_f_params["foo"] != composite.foo
