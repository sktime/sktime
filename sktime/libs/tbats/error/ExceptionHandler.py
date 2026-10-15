import warnings

from . import BatsWarning


class ExceptionHandler(object):

    def __init__(self, show_warnings=True):
        self.show_warnings = show_warnings

    def warn(self, message, warning_type=BatsWarning):
        if self.show_warnings:
            warnings.warn(message, warning_type)
            return True
        return False

    def exception(self, message, exception_type, previous_exception=None):
        if previous_exception is not None:
            raise exception_type(message) from previous_exception
        raise exception_type(message)
