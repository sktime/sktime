import warnings

from sktime.utils.warnings import _SuppressWarningPattern


def test_custom_showwarning_no_recursion():
    """Test that _custom_showwarning does not cause infinite recursion."""
    # Instantiate the class with a test pattern
    suppressor = _SuppressWarningPattern(FutureWarning, r"Test.*")

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always", category=FutureWarning)
        warnings.warn("Test warning", FutureWarning)
        assert len(w) == 1, "Warning should be emitted exactly once"
        assert "Test warning" in str(w[0].message)

    with suppressor:
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("Test warning", FutureWarning)
            assert len(w) == 0, "Warning should be suppressed"
