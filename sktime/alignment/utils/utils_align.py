"""Utility functions for handling and working with alignments."""

__author__ = ["fkiraly"]

import pandas as pd

# todo: need to wait for datatypes PR to merge
# from sktime.datatypes import check_is_mtype


def reindex_iloc(df, inds, copy=True):
    """Reindex pandas.DataFrame with iloc indices, potentially out of bound.

    Parameters
    ----------
    df: a pandas.DataFrame
    inds: iterable, list or pd.Series - iloc indices to reindex to
    copy: bool, optional, default=True - whether returned data frame is a new object
        if not, values are references; passed to copy arg of df.reindex

    Returns
    -------
    df_ret : pd.DataFrame - df reindexed to inds
        identical to df.iloc[inds] if inds contains no out of bound index
        out of bound indices will result in np.nan values
        entries are references to original DataFrame if copy=False

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.alignment.utils.utils_align import reindex_iloc
    >>> X = pd.DataFrame({'a' : [1,2,3,4]}, index=[-4,7,11,14])
    >>> reindex_iloc(X, [1, 2, 6])
         a
    1  2.0
    2  3.0
    6  NaN
    """
    df_ret = df.reset_index(drop=True).reindex(inds, copy=copy)

    return df_ret


def convert_align_to_align_loc(align, X, align_name="align", df_name="X", copy=True):
    """Convert iloc alignment to loc alignment, using reference data frame.

    Parameters
    ----------
    align: pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
        cols contain iloc index of X[i] mapped to alignment coordinate for alignment
    align_name: str, optional - name of "align" to display in error messages
    df_name: str, optional - name of "X" to display in error messages
    copy: bool, optional, default=True - whether returned data frame is a new object
        if not, values are references; passed to copy arg of df.reindex

    Returns
    -------
    pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
        cols contain loc index of X[i] mapped to alignment coordinate for alignment

    Examples
    --------
    align_df = pd.DataFrame({'ind0' : [1,2,3], 'ind1' : [0,2,4]})
    X = [pd.DataFrame({'a' : [1,2,3,4]}, index=[-4,7,11,14]),
            pd.DataFrame({'a' : [1,2,3,5,6]}, index=[4,8,12,16,20])
        ]

    convert_align_to_align_loc(align_df, X)
    """
    from sktime.datatypes import check_is_mtype

    check_is_mtype(
        align,
        "alignment",
        scitype="Alignment",
        var_name=align_name,
        msg_return_dict="list",
    )

    if not isinstance(X, list):
        raise ValueError(f"{df_name} must be a list of pandas.DataFrame")

    for Xi in X:
        if not isinstance(Xi, pd.DataFrame):
            raise ValueError(f"{df_name} must be a list of pandas.DataFrame")

    if copy:
        align = align.copy()

    if not len(X) == len(align.columns):
        raise ValueError(
            f"number of data frames in {df_name} must equal"
            f" number of index columns in {align_name}"
        )

    for i, Xi in enumerate(X):
        indi = "ind" + str(i)

        # reindex X to the alignment positions
        #  this also deals with np.nan indices
        loc_series = pd.Series(Xi.index).reindex(align[indi], copy=copy)
        align[indi] = loc_series.values

    return align
