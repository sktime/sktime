# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for singleton and multiton decorators."""

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.singleton import (
    _multiton,
    _registry,
    _singleton,
    clear_all_multitons,
    set_global_multiton_limit,
)


@_singleton
class _TestSingleton:
    def __init__(self, a=0):
        self.a = a

    def set_b(self, b):
        self.b = b


@_multiton
class _TestMultiton:
    def __init__(self, key, a=0):
        self.key = key
        self.a = a

    def set_b(self, b):
        self.b = b


@_multiton
class _TestMultitonOther:
    """A second, independent multiton class, for cross-class eviction tests."""

    def __init__(self, key, a=0):
        self.key = key
        self.a = a


@pytest.mark.skipif(
    not run_test_for_class(_singleton),
    reason="run test incrementally (if requested)",
)
def test_singleton():
    """Test singleton behaviour."""
    instance1 = _TestSingleton(42)
    instance1.set_b(43)
    instance2 = _TestSingleton()

    assert instance1 is instance2
    assert instance1.a == 42
    assert instance2.a == 42

    assert instance1.b == 43
    assert instance2.b == 43

    instance2.set_b(41)
    assert instance1.b == 41


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton():
    """Test multiton behaviour."""
    instance1 = _TestMultiton("key1", 42)
    instance1.set_b(43)
    instance2 = _TestMultiton("key2")
    instance2.set_b(44)

    instance1b = _TestMultiton("key1", 77)
    instance2b = _TestMultiton("key2")

    assert instance1 is not instance2
    assert instance1 is instance1b
    assert instance2 is instance2b

    assert instance1.a == 42
    assert instance2.a == 0
    assert instance1b.b == 43
    assert instance2.b == 44


@pytest.fixture
def _clean_multiton_registry():
    """Ensure the global multiton registry is unbounded and empty around a test."""
    clear_all_multitons()
    set_global_multiton_limit(None)
    yield
    clear_all_multitons()
    set_global_multiton_limit(None)


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton_default_is_unbounded(_clean_multiton_registry):
    """Without a limit set, multiton caching behaves as before (no eviction)."""
    keys = [f"k{i}" for i in range(5)]
    instances = [_TestMultiton(k) for k in keys]

    assert len(_registry) == 5
    for k, inst in zip(keys, instances):
        assert _TestMultiton(k) is inst


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton_global_limit_evicts_least_recently_used(_clean_multiton_registry):
    """Exceeding the global limit evicts the least recently used instance."""
    set_global_multiton_limit(2)

    a = _TestMultiton("a")
    _TestMultiton("b")
    assert len(_registry) == 2

    # adding a third distinct key evicts "a" (never re-accessed since creation)
    _TestMultiton("c")
    assert len(_registry) == 2

    # "a" was evicted, so requesting it again builds a genuinely new instance
    a_new = _TestMultiton("a")
    assert a_new is not a


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton_reaccess_protects_from_eviction(_clean_multiton_registry):
    """Re-accessing an instance marks it recently used, protecting it from eviction."""
    set_global_multiton_limit(2)

    x = _TestMultiton("x")
    y = _TestMultiton("y")
    _TestMultiton("x")  # touch x again -> x is now more recently used than y

    _TestMultiton("z")  # forces an eviction: y should go, not x

    assert _TestMultiton("x") is x
    assert _TestMultiton("y") is not y


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_multiton_global_limit_shared_across_classes(_clean_multiton_registry):
    """The global limit is shared across different multiton-decorated classes."""
    set_global_multiton_limit(2)

    _TestMultiton("p")
    _TestMultitonOther("q")
    assert len(_registry) == 2

    # a new instance of either class can evict an instance of the other class
    _TestMultiton("r")
    assert len(_registry) == 2


@pytest.mark.skipif(
    not run_test_for_class(_multiton),
    reason="run test incrementally (if requested)",
)
def test_clear_all_multitons_removes_every_instance(_clean_multiton_registry):
    """clear_all_multitons empties the registry and the owning per-class caches."""
    a = _TestMultiton("a")
    _TestMultitonOther("b")
    assert len(_registry) == 2

    clear_all_multitons()
    assert len(_registry) == 0

    assert _TestMultiton("a") is not a
