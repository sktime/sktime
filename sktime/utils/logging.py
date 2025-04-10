"""Logging functionality for sktime."""

import functools
import logging
import traceback

__author__ = ["Ankit-1204"]

logger = logging.getLogger(__name__)


def log_exceptions(method):
    """Log method calls and exceptions with full context."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        method_name = method.__name__
        cls_name = self.__class__.__name__
        try:
            result = method(self, *args, **kwargs)
            logger.info(f"[{cls_name}] << {method_name} completed.")
            return result
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"[{cls_name}] !! {method_name} failed: {e}\n{tb_str}")
            raise RuntimeError(f"[{cls_name}] {method_name} failed.") from e

    return wrapper
