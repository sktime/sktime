# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseObject universal base class.

tests in this module:

    test_get_class_tags  - tests get_class_tags inheritance logic
    test_get_class_tag   - tests get_class_tag logic, incl default value
    test_get_tags        - tests get_tags inheritance logic
    test_get_tag         - tests get_tag logic, incl default value
    test_set_tags        - tests set_tags logic and related get_tags inheritance

    test_reset           - tests reset logic on a simple, non-composite estimator
    test_reset_composite - tests reset logic on a composite estimator

    test_components         - tests retrieval of list of components via _components
    test_get_fitted_params  - tests get_fitted_params logic, nested and non-nested

    test_eq_dunder       - tests __eq__ dunder to compare parameter definition
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
    "test_components",
    "test_param_alias",
    "test_nested_set_params_and_alias",
    "test_get_fitted_params",
    "test_eq_dunder",
]

from copy import deepcopy

import pytest

from sktime.base import BaseEstimator, BaseObject
from sktime.utils.dependencies import _check_soft_dependencies


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

# default tags in BaseObject
DEFAULT_TAGS = BaseObject._tags


def test_get_class_tags():
    """Tests get_class_tags class method of BaseObject for correctness.

    Raises
    ------
    AssertError if inheritance logic in get_class_tags is incorrect
    """
    child_tags = FIXTURE_CLASSCHILD.get_class_tags()

    msg = "Inheritance logic in BaseObject.get_class_tags is incorrect"

    expected_tags = FIXTURE_CLASSCHILD_TAGS.copy()
    expected_tags.update(DEFAULT_TAGS)
    assert child_tags == expected_tags, msg


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

    expected_tags = FIXTURE_OBJECT_TAGS.copy()
    expected_tags.update(DEFAULT_TAGS)
    assert object_tags == expected_tags, msg


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

    expected_tags = FIXTURE_OBJECT_SET_TAGS.copy()
    expected_tags.update(DEFAULT_TAGS)
    assert FIXTURE_OBJECT_SET.get_tags() == expected_tags, msg


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
    assert set(comp_comps.keys()) == {"foo_"}
    assert comp_comps["foo_"] is composite.foo_
    assert comp_comps["foo_"] is not composite.foo


class AliasTester(BaseObject):
    def __init__(self, a, bar=42):
        self.a = a
        self.bar = bar


@pytest.mark.skipif(
    _check_soft_dependencies("skbase<0.6.1", severity="none"),
    reason="aliasing was introduced in skbase 0.6.1",
)
def test_param_alias():
    """Tests parameter aliasing with parameter string shorthands.

    Raises
    ------
    AssertionError if parameters that should be set via __ are not set
    AssertionError if error that should be raised is not raised
    """
    non_composite = AliasTester(a=42, bar=4242)
    composite = CompositionDummy(foo=non_composite)

    # this should write to a of foo, because there is only one suffix called a
    composite.set_params(**{"a": 424242})
    assert composite.get_params()["foo__a"] == 424242

    # this should write to bar of composite, because "bar" is a full parameter string
    #   there is a suffix in foo, but if the full string is there, it writes to that
    composite.set_params(**{"bar": 424243})
    assert composite.get_params()["bar"] == 424243

    # trying to write to bad_param should raise an exception
    # since bad_param is neither a suffix nor a full parameter string
    with pytest.raises(ValueError, match=r"Invalid parameter keys provided to"):
        composite.set_params(**{"bad_param": 424242})

    # new example: highly nested composite with identical suffixes
    non_composite1 = composite
    non_composite2 = AliasTester(a=42, bar=4242)
    uber_composite = CompositionDummy(foo=non_composite1, bar=non_composite2)

    # trying to write to a should raise an exception
    # since there are two suffix a, and a is not a full parameter string
    with pytest.raises(ValueError, match=r"does not uniquely determine parameter key"):
        uber_composite.set_params(**{"a": 424242})

    # same as above, should overwrite "bar" of uber_composite
    uber_composite.set_params(**{"bar": 424243})
    assert uber_composite.get_params()["bar"] == 424243


@pytest.mark.skipif(
    _check_soft_dependencies("skbase<0.6.1", severity="none"),
    reason="aliasing was introduced in skbase 0.6.1",
)
def test_nested_set_params_and_alias():
    """Tests that nested param setting works correctly.

    This specifically tests that parameters of components can be provided,
    even if that component is not present in the object that set_params is called on,
    but is also being set in the same set_params call.

    Also tests alias resolution, using recursive end state after set_params.

    Raises
    ------
    AssertionError if parameters that should be set via __ are not set
    AssertionError if error that should be raised is not raised
    """
    non_composite = AliasTester(a=42, bar=4242)
    composite = CompositionDummy(foo=0)

    # this should write to a of foo
    # potential error here is that composite does not have foo__a to start with
    # so error catching or writing foo__a to early could cause an exception
    composite.set_params(**{"foo": non_composite, "foo__a": 424242})
    assert composite.get_params()["foo__a"] == 424242

    non_composite = AliasTester(a=42, bar=4242)
    composite = CompositionDummy(foo=0)

    # same, and recognizing that foo__a is the only matching suffix in the end state
    composite.set_params(**{"foo": non_composite, "a": 424242})
    assert composite.get_params()["foo__a"] == 424242

    # new example: highly nested composite with identical suffixes
    non_composite1 = composite
    non_composite2 = AliasTester(a=42, bar=4242)
    uber_composite = CompositionDummy(foo=42, bar=42)

    # trying to write to a should raise an exception
    # since there are two suffix a, and a is not a full parameter string
    with pytest.raises(ValueError, match=r"does not uniquely determine parameter key"):
        uber_composite.set_params(
            **{"a": 424242, "foo": non_composite1, "bar": non_composite2}
        )

    uber_composite = CompositionDummy(foo=non_composite1, bar=42)

    # same as above, should overwrite "bar" of uber_composite
    uber_composite.set_params(**{"bar": 424243})
    assert uber_composite.get_params()["bar"] == 424243


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
    comp_f_params_shallow = composite.get_fitted_params(deep=False)

    assert isinstance(non_comp_f_params, dict)
    assert set(non_comp_f_params.keys()) == {"foo"}

    assert isinstance(comp_f_params, dict)
    assert set(comp_f_params) == {"foo", "foo__foo"}
    assert set(comp_f_params_shallow) == {"foo"}
    assert comp_f_params["foo"] is composite.foo_
    assert comp_f_params["foo"] is not composite.foo
    assert comp_f_params_shallow["foo"] is composite.foo_
    assert comp_f_params_shallow["foo"] is not composite.foo


class ConfigTester(BaseObject):
    _config = {"foo_config": 42, "bar": "a"}

    clsvar = 210

    def __init__(self, a, b=42):
        self.a = a
        self.b = b
        self.c = 84


def test_set_get_config():
    """Test logic behind get_config, set_config.

    Raises
    ------
    AssertionError if logic behind get_config, set_config is incorrect, logic tested:
        calling get_fitted_params on a non-composite fittable returns the fitted param
        calling get_fitted_params on a composite returns all nested params
    """
    # get default config dict
    base_config = BaseObject().get_config()
    base_keys = set(base_config.keys())

    obj = ConfigTester(4242)

    config_start = obj.get_config()
    assert isinstance(config_start, dict)
    expected_config_start_keys = {"foo_config", "bar"}.union(base_keys)
    assert set(config_start.keys()) == expected_config_start_keys
    assert config_start["foo_config"] == 42
    assert config_start["bar"] == "a"

    setconfig_return = obj.set_config(foobar=126)
    assert obj is setconfig_return

    obj.set_config(**{"bar": "b"})
    config_end = obj.get_config()
    assert isinstance(config_end, dict)
    expected_config_end_keys = {"foo_config", "bar", "foobar"}.union(base_keys)
    assert set(config_end.keys()) == expected_config_end_keys
    assert config_end["foo_config"] == 42
    assert config_end["bar"] == "b"
    assert config_end["foobar"] == 126


def test_eq_dunder():
    """Tests equality dunder for BaseObject descendants.

    Equality should be determined only by get_params results.

    Raises
    ------
    AssertionError if logic behind __eq__ is incorrect, logic tested:
        equality of non-composites depends only on params, not on identity
        equality of composites depends only on params, not on identity
        result is not affected by fitting the estimator
    """
    non_composite = FittableCompositionDummy(foo=42)
    non_composite_2 = FittableCompositionDummy(foo=42)
    non_composite_3 = FittableCompositionDummy(foo=84)

    composite = FittableCompositionDummy(foo=non_composite)
    composite_2 = FittableCompositionDummy(foo=non_composite_2)
    composite_3 = FittableCompositionDummy(foo=non_composite_3)

    assert non_composite == non_composite
    assert composite == composite
    assert non_composite == non_composite_2
    assert non_composite != non_composite_3
    assert non_composite_2 != non_composite_3
    assert composite == composite_2
    assert composite != composite_3
    assert composite_2 != composite_3

    # equality should not be affected by fitting
    composite.fit()
    non_composite_2.fit()

    assert non_composite == non_composite
    assert composite == composite
    assert non_composite == non_composite_2
    assert non_composite != non_composite_3
    assert non_composite_2 != non_composite_3
    assert composite == composite_2
    assert composite != composite_3
    assert composite_2 != composite_3
