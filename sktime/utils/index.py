"""Utility functions for working with indices."""

import hashlib

import numpy as np
import pandas as pd

# fixed key for the pandas element hash, so that digests do not depend on
# the pandas default, which is a documented constant but not a guaranteed one
_FOLD_HASH_KEY = "sktime_fold_hash"

# number of hex characters kept from the sha256 digest
_FOLD_HASH_CHARS = 16


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


def _get_index(obj):
    """Coerce a fold data container to the index identifying its rows.

    Parameters
    ----------
    obj : pd.Index, pd.Series, pd.DataFrame, or 1D np.ndarray

    Returns
    -------
    pd.Index, or None if no index can be obtained from ``obj``
    """
    if isinstance(obj, pd.Index):
        return obj
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.index
    if isinstance(obj, np.ndarray) and obj.ndim == 1:
        return pd.Index(obj)
    return None


def fold_fingerprint(*fold_data):
    """Compute a stable digest of the indices making up a single fold.

    Two folds have the same fingerprint if and only if, up to hash collision,
    every positional argument carries the same index, in the same order.
    The fingerprint depends only on the indices, not on the values, and not on
    the estimator that the fold was used with. Two ``evaluate`` calls that
    produce the same folds therefore produce the same fingerprints, and two
    calls that do not produce the same folds do not.

    The digest is computed with ``hashlib``, over element hashes obtained from
    ``pandas.util.hash_pandas_object`` with a fixed hash key. It is therefore
    stable across processes and across runs, unlike the built-in ``hash``,
    which is salted per process for ``str`` and ``bytes``.

    Cost is linear in the total number of index elements. Fold data values are
    never touched, so the cost does not grow with the number of columns or with
    the size of the observations.

    Parameters
    ----------
    *fold_data : pd.Index, pd.Series, pd.DataFrame, or 1D np.ndarray
        Containers whose indices jointly identify the fold, in a fixed order.
        Positions are part of the digest, so passing train and test data in the
        other order gives a different fingerprint.

    Returns
    -------
    fingerprint : str, or None
        16 hex characters, or None if any argument has no index that can be
        obtained or hashed. None means "not comparable", and callers should
        treat it as such rather than as a fold identity.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.utils.index import fold_fingerprint
    >>> y = pd.Series(range(6))
    >>> fold_fingerprint(y[:4], y[4:]) == fold_fingerprint(y[:4], y[4:])
    True
    >>> fold_fingerprint(y[:4], y[4:]) == fold_fingerprint(y[:3], y[3:])
    False
    """
    hasher = hashlib.sha256()

    for pos, obj in enumerate(fold_data):
        idx = _get_index(obj)
        if idx is None:
            return None
        try:
            hashed = pd.util.hash_pandas_object(
                idx, index=False, categorize=False, hash_key=_FOLD_HASH_KEY
            )
            # cast to a fixed width and byte order, so that the digest does not
            # depend on the platform the evaluation was run on
            payload = np.asarray(hashed, dtype=np.uint64).astype("<u8").tobytes()
        except Exception:
            # an index that cannot be hashed is not a fold identity, and this
            # is a bookkeeping column, so it must not break the evaluation
            return None
        # length and position are part of the digest, so that moving elements
        # between two folds, or swapping the folds, changes the fingerprint
        hasher.update(f"{pos}:{len(idx)}:".encode())
        hasher.update(payload)

    return hasher.hexdigest()[:_FOLD_HASH_CHARS]
