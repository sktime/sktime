"""Utility functions for working with indices."""

import numpy as np


def random_ss_ix(ix, size, replace=True):
    """Randomly uniformly sample indices from a list of indices.

    Parameters
    ----------
    ix : pd.Index or subsettable iterable via getitem
        list of indices to sample from
    size : int
        number of indices to sample
    replace : bool, default=True
        whether to sample with replacement
    """
    a = range(len(ix))
    ixs = ix[np.random.choice(a, size=size, replace=replace)]
    return ixs
