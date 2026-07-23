"""Tests for benchmarking base classes."""

import pytest

from sktime.benchmarking.base import BaseResults


def test_base_results_save_raises_not_implemented_error():
    """Test that BaseResults.save is not silently ignored."""
    results = BaseResults()

    with pytest.raises(NotImplementedError):
        results.save()
