"""Utility to find all closest elements in sorted list b to sorted list a."""


def _find_closest_elements(a, b):
    """Find the closest element in b for each element in a.

    Parameters
    ----------
    a : 1D array-like
        An ordered (sorted) list of elements.
    b : 1D array-like
        Another ordered (sorted) list of elements.

    Returns
    -------
    closest : list, same length as a
        a list of closest elements in ``b`` for each element in ``a``.
        In case of ties, the first closest element is chosen.

    Examples
    --------
    >>> from sktime.performance_metrics.detection.utils._closest import (
    ...     _find_closest_elements
    ... )
    >>> a = [1, 3, 5]
    >>> b = [2, 3.1, 3.2, 4, 6]
    >>> pointer = DirectedHausdorff()._find_closest_elements(a, b)
    """
    # Pointers for traversing A and B
    i, j = 0, 0
    n, m = len(a), len(b)

    # List to store the result
    result = []

    while i < n:
        # Move pointer j in b to get the closest value to a[i]
        while j + 1 < m and abs(b[j + 1] - a[i]) < abs(b[j] - a[i]):
            j += 1

        # Append the closest value in b for a[i]
        result.append(b[j])

        # Move to the next element in a
        i += 1

    return result
