# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the dynamic compile-on-demand utility."""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest


def _compiler_available():
    """Check if a C compiler is present without polluting module scope."""
    try:
        from sktime.utils.dependencies.compile import has_compiler

        return has_compiler()
    except Exception:
        return False


def test_has_compiler_returns_bool():
    """Test that has_compiler returns a boolean value."""
    from sktime.utils.dependencies.compile import has_compiler

    result = has_compiler()
    assert isinstance(result, bool)


def test_import_or_compile_extension_loads_cached_module():
    """Test import_or_compile_extension for an already-importable module."""
    from sktime.utils.dependencies.compile import import_or_compile_extension

    # Use a module that is guaranteed to be importable (stdlib)
    mod = import_or_compile_extension(
        module_name="json",
        pyx_relative_path="dummy.pyx",
        parent_file_path=__file__,
    )
    assert mod is importlib.import_module("json")


@pytest.mark.skipif(
    not _compiler_available(),
    reason="No working C compiler available in the environment",
)
def test_import_or_compile_extension_compiles_pyx():
    """Test that import_or_compile_extension compiles a .pyx file dynamically."""
    from sktime.utils.dependencies.compile import import_or_compile_extension

    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_name = "_test_compile_ext.pyx"
        pyx_path = Path(tmpdir) / pyx_name

        # Minimal Cython source that defines a callable function
        pyx_path.write_text("def add_numbers(int a, int b):\n    return a + b\n")

        mod = import_or_compile_extension(
            module_name="_test_compile_ext",
            pyx_relative_path=pyx_name,
            parent_file_path=str(pyx_path),
        )

        assert hasattr(mod, "add_numbers")
        assert mod.add_numbers(3, 4) == 7

        # Clean up sys.path and sys.modules
        if "_test_compile_ext" in sys.modules:
            del sys.modules["_test_compile_ext"]


def test_import_or_compile_extension_raises_without_compiler():
    """Test that a clear ImportError is raised when no compiler is available."""
    from sktime.utils.dependencies.compile import has_compiler

    if has_compiler():
        pytest.skip("Compiler is available; cannot test missing-compiler path")

    from sktime.utils.dependencies.compile import import_or_compile_extension

    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_name = "_test_no_compiler.pyx"
        pyx_path = Path(tmpdir) / pyx_name
        pyx_path.write_text("def dummy(): pass\n")

        with pytest.raises(ImportError, match="No working C compiler"):
            import_or_compile_extension(
                module_name="_test_no_compiler",
                pyx_relative_path=pyx_name,
                parent_file_path=str(pyx_path),
            )
