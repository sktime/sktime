"""Utility functions for dataset generation."""


def recycle_list(lst: list, n: int) -> list:
    """Recycle a list to ensure it has exactly `n` elements.

    Repeats the elements of the list until it reaches the desired length.
    If the list is longer than `n`, it truncates it to the first `n` elements.

    Parameters
    ----------
    lst : list
        The list to recycle.
    n : int
        The desired length of the output list.

    Returns
    -------
    list
        A list with exactly `n` elements, recycled from the input list if necessary.
    """
    if len(lst) == 0:
        raise ValueError("The list cannot be empty.")

    if len(lst) < n:
        return lst * (n // len(lst)) + lst[: n % len(lst)]

    return lst[:n]
