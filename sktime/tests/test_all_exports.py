# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test that all __all__ declarations in the codebase contain only strings."""

import ast
from pathlib import Path

import pytest

_SKTIME_ROOT = Path(__file__).parent.parent


def test_all_exports_are_strings():
    """Test that all elements in __all__ lists across the codebase are strings.
    
    This prevents TypeErrors when users attempt to perform a wildcard import
    (e.g., `from sktime.module import *`), which strictly requires `__all__`
    to be a sequence of strings in Python.
    """
    violations = []

    for py_file in _SKTIME_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Look for assignments to __all__
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        # We found __all__ = ...
                        if isinstance(node.value, ast.List) or isinstance(node.value, ast.Tuple):
                            for element in node.value.elts:
                                # In Python 3.8+, string literals are ast.Constant
                                # In older versions, they are ast.Str
                                if isinstance(element, ast.Constant):
                                    if not isinstance(element.value, str):
                                        violations.append((py_file, element))
                                elif hasattr(ast, "Str") and isinstance(element, ast.Str):
                                    pass # valid string
                                else:
                                    violations.append((py_file, element))

    if violations:
        msg = "Found __all__ declarations containing non-string elements (which breaks wildcard imports):\n"
        for py_file, element in violations:
            rel_path = py_file.relative_to(_SKTIME_ROOT)
            msg += f"- {rel_path} (Line {element.lineno})\n"
        pytest.fail(msg)
