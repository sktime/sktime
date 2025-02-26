import warnings

from sktime.utils.warnings import _SuppressWarningPattern


def test_custom_showwarning_no_recursion():
    """Test that _custom_showwarning does not cause infinite recursion."""
    # Instantiate the class with a test pattern
    suppressor = _SuppressWarningPattern(FutureWarning, r"Test.*")

    # Monkey-patch temporário para usar o _custom_showwarning
    original_showwarning = warnings.showwarning
    warnings.showwarning = suppressor._custom_showwarning

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always", category=FutureWarning)
        warnings.warn("Test warning", FutureWarning)
        assert len(w) == 1, "Warning should be emitted exactly once"
        assert "Test warning" in str(w[0].message)

    # Restore the original handler
    warnings.showwarning = original_showwarning
