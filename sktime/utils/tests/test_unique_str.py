"""Tests for unique string utilities."""

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.unique_str import _make_strings_unique


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
class TestMakeStringsUnique:
    """Tests for _make_strings_unique function."""

    def test_new_str_already_unique(self):
        """Test that a unique string is returned unchanged."""
        result = _make_strings_unique(["a", "b", "c"], "d")
        assert result == "d"

    def test_new_str_conflicts_once(self):
        """Test that a conflicting string gets _2 appended."""
        result = _make_strings_unique(["a", "b", "c"], "a")
        assert result == "a_2"

    def test_new_str_conflicts_multiple_times(self):
        """Test that a string conflicting with existing _2 gets _4 appended.

        Counter logic: first call appends _2 (clashes), recursive call
        increments counter to 4 and appends _4 (free).
        """
        result = _make_strings_unique(["a", "a_2", "b"], "a")
        assert result == "a_4"

    def test_new_str_conflicts_three_times(self):
        """Test the chain a -> a_2 -> a_3 -> a_4."""
        result = _make_strings_unique(["a", "a_2", "a_3"], "a")
        assert result == "a_4"

    def test_empty_list(self):
        """Test with an empty list (no conflicts)."""
        result = _make_strings_unique([], "x")
        assert result == "x"

    def test_non_conflicting_similar_strings(self):
        """Test that similar but non-conflicting strings pass through."""
        result = _make_strings_unique(["ab", "abc"], "a")
        assert result == "a"
