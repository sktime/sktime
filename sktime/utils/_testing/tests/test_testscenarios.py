#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["fkiraly"]
__all__ = []


from sktime.utils._testing.scenarios import TestScenario


class TestedMockClass:
    """Mock class to test TestScenario."""

    def __init__(self, a):
        self.a = a

    def foo(self, b):
        """Test method for mock class to test TestScenario."""
        self.a += b
        return self.a

    def bar(self, c, d="0"):
        """Test method for mock class to test TestScenario."""
        self.a += c
        self.a += d
        return self.a

    @classmethod
    def baz(cls):
        return "foo"


def test_testscenario_object_args_only():
    """Test basic workflow: construct only with args, call run with minimal args."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={"foo": {"b": "cali"}, "bar": {"c": "fragi", "d": "listic"}}
    )

    result = scenario.run(obj, method_sequence=["foo", "bar"])

    assert result == "supercalifragilistic"


def test_testscenario_object_default_method_sequence():
    """Test basic workflow: construct with args and default method sequence."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={"foo": {"b": "cali"}, "bar": {"c": "fragi", "d": "listic"}},
        default_method_sequence=["foo", "bar"],
    )

    result = scenario.run(obj)

    assert result == "supercalifragilistic"


def test_testscenario_object_default_arg_sequence():
    """Test basic workflow: construct with args and default arg sequence."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={"foo": {"b": "cali"}, "bar": {"c": "fragi", "d": "listic"}},
        default_arg_sequence=["foo", "bar"],
    )

    result = scenario.run(obj)

    assert result == "supercalifragilistic"


def test_testscenario_object_return_all():
    """Test basic workflow: construct with args and default arg sequence."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={"foo": {"b": "cali"}, "bar": {"c": "fragi", "d": "listic"}},
        default_arg_sequence=["foo", "bar"],
    )

    result = scenario.run(obj, return_all=True)

    assert result == ["supercali", "supercalifragilistic"]


def test_testscenario_object_multi_call_defaults():
    """Test basic workflow: default args where methods are called multiple times."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={
            "foo": {"b": "cali"},
            "bar": {"c": "fragi", "d": "listic"},
            "foo-2nd": {"b": "expi"},
            "bar-2nd": {"c": "ali", "d": "docious"},
        },
        default_arg_sequence=["foo", "bar", "foo-2nd", "bar-2nd"],
        default_method_sequence=["foo", "bar", "foo", "bar"],
    )

    result = scenario.run(obj)

    assert result == "supercalifragilisticexpialidocious"


def test_testscenario_object_multi_call_in_run():
    """Test advanced workflow: run args where methods are called multiple times."""
    obj = TestedMockClass(a="super")
    scenario = TestScenario(
        args={
            "foo": {"b": "cali"},
            "bar": {"c": "fragi", "d": "listic"},
            "foo-2nd": {"b": "expi"},
            "bar-2nd": {"c": "ali", "d": "docious"},
        },
    )

    result = scenario.run(
        obj,
        arg_sequence=["foo", "bar", "foo-2nd", "bar-2nd"],
        method_sequence=["foo", "bar", "foo", "bar"],
    )

    assert result == "supercalifragilisticexpialidocious"


def test_testscenario_class_full_options():
    """Test advanced workflow: constructor and methods called multiple times."""
    obj = TestedMockClass
    scenario = TestScenario(
        args={
            "__init__": {"a": "super"},
            "foo": {"b": "cali"},
            "bar": {"c": "fragi", "d": "listic"},
            "foo-2nd": {"b": "expi"},
            "bar-2nd": {"c": "ali", "d": "docious"},
        },
    )

    result = scenario.run(
        obj,
        arg_sequence=["__init__", "foo", "bar", "foo-2nd", "bar-2nd"],
        method_sequence=["__init__", "foo", "bar", "foo", "bar"],
    )

    assert result == "supercalifragilisticexpialidocious"


def test_testscenario_class_simple():
    """Test advanced workflow: constructor, but only simple function calls."""
    obj = TestedMockClass
    scenario = TestScenario(
        args={
            "__init__": {"a": "super"},
            "foo": {"b": "cali"},
            "bar": {"c": "fragi", "d": "listic"},
        },
    )

    result = scenario.run(
        obj,
        method_sequence=["__init__", "foo", "bar"],
    )

    assert result == "supercalifragilistic"
