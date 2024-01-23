"""String-to-int coercion utility strtobool formerly in deprecated distutils."""


# distutils has been removed in Python 3.12,
# but this function is still used in parts of the sktime codebase
def strtobool(val):
    """Convert a string representation of truth to int, true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Parameters
    ----------
    val : str
        A string representation of truth.

    Returns
    -------
    int, 0 or 1
        val coerced to int
        1 if val is a true value, 0 if val is a false value (see above)

    Raises
    ------
    ValueError
        If val is anything other than the strings above.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")
