# -*- coding: utf-8 -*-
"""Base utils and classes for Mock Estimators."""

from copy import deepcopy


class MockEstimatorMixin:
    """Mixin class for constructing Mock estimators."""

    @property
    def log(self):
        """Log of the methods called and the parameters passed in each method."""
        return self._log

    def _update_log(self, method_name, method_kargs):
        self._log.append((method_name, deepcopy(method_kargs)))
