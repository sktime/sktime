"""Doctest utilities."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import contextlib
import doctest
import io


def run_doctest(
    f,
    verbose=False,
    name=None,
    compileflags=None,
    optionflags=doctest.ELLIPSIS,
    raise_on_error=True,
):
    """Run doctests for a given function or class, and return or raise.

    Parameters
    ----------
    f : callable
        Function or class to run doctests for.
    verbose : bool, optional (default=False)
        If True, print the results of the doctests.
    name : str, optional (default=f.__name__, if available, otherwise "NoName")
        Name of the function or class.
    compileflags : int, optional (default=None)
        Flags to pass to the Python parser.
    optionflags : int, optional (default=doctest.ELLIPSIS)
        Flags to control the behaviour of the doctest.
    raise_on_error : bool, optional (default=True)
        If True, raise an exception if the doctests fail.

    Returns
    -------
    doctest_output : str
        Output of the doctests.

    Raises
    ------
    RuntimeError
        If raise_on_error=True and the doctests fail.
    """
    doctest_output_io = io.StringIO()
    with contextlib.redirect_stdout(doctest_output_io):
        doctest.run_docstring_examples(
            f=f,
            globs=globals(),
            verbose=verbose,
            name=name,
            compileflags=compileflags,
            optionflags=optionflags,
        )
    doctest_output = doctest_output_io.getvalue()

    if name is None:
        name = f.__name__ if hasattr(f, "__name__") else "NoName"

    if raise_on_error and len(doctest_output) > 0:
        raise RuntimeError(
            f"Docstring examples failed doctests "
            f"for {name}, doctest output: {doctest_output}"
        )
    return doctest_output
