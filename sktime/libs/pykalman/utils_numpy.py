"""Utility functions to handle numpy 2."""

from skbase.utils.dependencies import _check_soft_dependencies

numpy2 = _check_soft_dependencies("numpy>=2", severity="none")


def newbyteorder(arr, new_order):
    """Change the byte order of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    new_order : str
        Byte order to force.

    Returns
    -------
    arr : ndarray
        Array with new byte order.
    """
    if numpy2:
        return arr.view(arr.dtype.newbyteorder(new_order))
    else:
        return arr.newbyteorder(new_order)
