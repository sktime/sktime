"""String utilities for making strings unique."""


def _make_strings_unique(strlist, new_str, counter=1):
    """Make a list or tuple of strings unique by appending _int of occurrence.

    Parameters
    ----------
    strlist : list of unique strings
    new_str : str
        string to make unique

    Returns
    -------
    str
        new string with _int appended if not unique
    """
    # Note default counter is 1 since the second element in the list
    # should be appended with _2
    counter += 1
    assert len(strlist) == len(set(strlist)), "strlist must be unique"
    if new_str in strlist:
        if counter == 2:
            # In that case, we assume that no string was appended yet.
            new_str += f"_{counter}"
        else:
            new_str = "".join(new_str.split("_")[:-1]) + f"_{counter}"
        return _make_strings_unique(strlist, new_str, counter + 1)
    return new_str
