# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for singleton and multiton decorators."""

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.singleton import _detach_from_multiton, _multiton, _singleton


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


@pytest.mark.skipif(
    not run_test_for_class(_detach_from_multiton),
    reason="run test incrementally (if requested)",
)
def test_detach_from_multiton():
    """Test that a detached multiton instance is independent of the multiton."""
    instance = _TestMultiton("key_detach", 42)
    instance.set_b([1, 2])

    detached = _detach_from_multiton(instance)
    detached.b.append(3)
    detached.a = 0

    assert detached is not instance
    assert instance.a == 42
    assert instance.b == [1, 2]
    assert _TestMultiton("key_detach") is instance


def test_detach_from_multiton_preserves_shared_torch_weights():
    """Training a detached model changes neither cached parameters nor buffers."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    model.register_buffer("running_count", torch.tensor(0.0))
    model.eval()
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    detached = _detach_from_multiton(model)
    assert not detached.training
    assert detached.weight.data_ptr() != model.weight.data_ptr()
    assert detached.running_count.data_ptr() != model.running_count.data_ptr()

    optimizer = torch.optim.SGD(detached.parameters(), lr=0.1)
    optimizer.zero_grad()
    detached(torch.ones(3, 2)).sum().backward()
    optimizer.step()
    detached.running_count.add_(1)

    assert not torch.equal(detached.weight, model.weight)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
