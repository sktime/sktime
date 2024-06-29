"""Utility functions for adapting to sklearn."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np


def prep_skl_df(df, copy_df=False):
    """Make df compatible with sklearn input expectations.

    Changes:
    turns column index into a list of strings

    Parameters
    ----------
    df : pd.DataFrame
        list of indices to sample from
    copy_df : bool, default=False
        whether to mutate df or return a copy
        if False, index of df is mutated
        if True, original df is not mutated. If index is not a list of strings,
        a copy is made and the copy is mutated. Otherwise, the original df is returned.
    """
    cols = df.columns
    str_cols = cols.astype(str)

    if not np.all(str_cols == cols):
        if copy_df:
            df = df.copy()
        df.columns = str_cols

    return df
