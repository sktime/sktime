"""Context manager to suppress stdout and stderr.

With lower skbase versions,
there might not be skbase.utils.stdout_mute and skbase.utils.stderr_mute.
"""

__author__ = ["fkiraly", "XinyuWu"]

import io
import sys


class _StdoutMute:
    """A context manager to suppress stdout.

    Exception handling on exit can be customized by overriding
    the ``_handle_exit_exceptions`` method.

    Parameters
    ----------
    active : bool, default=True
        Whether to suppress stdout or not.
        If True, stdout is suppressed.
        If False, stdout is not suppressed, and the context manager does nothing
        except catch and suppress ModuleNotFoundError.
    """

    def __init__(self, active=True):
        self.active = active

    def __enter__(self):
        """Context manager entry point."""
        # capture stdout if active
        # store the original stdout so it can be restored in __exit__
        if self.active:
            self._stdout = sys.stdout
            sys.stdout = io.StringIO()

    def __exit__(self, type, value, traceback):  # noqa: A002
        """Context manager exit point."""
        # restore stdout if active
        # if not active, nothing needs to be done, since stdout was not replaced
        if self.active:
            sys.stdout = self._stdout

        if type is not None:
            return self._handle_exit_exceptions(type, value, traceback)

        # if no exception was raised, return True to indicate successful exit
        # return statement not needed as type was None, but included for clarity
        return True

    def _handle_exit_exceptions(self, type, value, traceback):  # noqa: A002
        """Handle exceptions raised during __exit__.

        Parameters
        ----------
        type : type
            The type of the exception raised.
            Known to be not-None and Exception subtype when this method is called.
        value : Exception
            The exception instance raised.
        traceback : traceback
            The traceback object associated with the exception.
        """
        # by default, all exceptions are raised
        return False


class _StderrMute:
    """A context manager to suppress stderr.

    Exception handling on exit can be customized by overriding
    the ``_handle_exit_exceptions`` method.

    Parameters
    ----------
    active : bool, default=True
        Whether to suppress stderr or not.
        If True, stderr is suppressed.
        If False, stderr is not suppressed, and the context manager does nothing
        except catch and suppress ModuleNotFoundError.
    """

    def __init__(self, active=True):
        self.active = active

    def __enter__(self):
        """Context manager entry point."""
        # capture stderr if active
        # store the original stderr so it can be restored in __exit__
        if self.active:
            self._stderr = sys.stderr
            sys.stderr = io.StringIO()

    def __exit__(self, type, value, traceback):  # noqa: A002
        """Context manager exit point."""
        # restore stderr if active
        # if not active, nothing needs to be done, since stderr was not replaced
        if self.active:
            sys.stderr = self._stderr

        if type is not None:
            return self._handle_exit_exceptions(type, value, traceback)

        # if no exception was raised, return True to indicate successful exit
        # return statement not needed as type was None, but included for clarity
        return True

    def _handle_exit_exceptions(self, type, value, traceback):  # noqa: A002
        """Handle exceptions raised during __exit__.

        Parameters
        ----------
        type : type
            The type of the exception raised.
            Known to be not-None and Exception subtype when this method is called.
        value : Exception
            The exception instance raised.
        traceback : traceback
            The traceback object associated with the exception.
        """
        # by default, all exceptions are raised
        return False
