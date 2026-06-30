# Intentional cross-module import to demonstrate test_cross_module_imports.py catches it.
# This file exists only to show the guard works — see PR #10479.
from sktime.classification.base import BaseClassifier  # noqa: F401
