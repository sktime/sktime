"""Minimal tsbootstrap-style bootstrap implementations.

The vendored classes here intentionally implement only the subset of behavior
used by sktime adapters and tests.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["sktime developers"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.base import BaseObject


def _sample_moving_block_indices(
    n_obs, block_length, replacement, random_state, size=None
):
    """Sample integer indices using moving-block logic."""
    if size is None:
        size = n_obs

    rng = check_random_state(random_state)

    if n_obs <= 1:
        return np.zeros(size, dtype=int)

    if block_length is None:
        block_length = 1
    block_length = max(1, min(int(block_length), n_obs - 1))

    if block_length == 1 and not replacement:
        # If a full permutation is requested, sample without replacement.
        if size == n_obs:
            idx = np.arange(n_obs)
            rng.shuffle(idx)
            return idx
        return rng.choice(np.arange(n_obs), size=size, replace=False)

    if block_length == 1:
        return rng.choice(np.arange(n_obs), size=size, replace=True)

    total_num_blocks = int(np.ceil(size / block_length)) + 1
    block_origins = rng.choice(
        n_obs - block_length + 1, size=total_num_blocks, replace=replacement
    )
    idx = [j for origin in block_origins for j in range(origin, origin + block_length)]
    remove_first = rng.choice(block_length)
    idx = idx[remove_first : remove_first + size]
    return np.asarray(idx, dtype=int)


class _BaseBootstrap(BaseObject):
    """Base class for vendored tsbootstrap-like bootstrappers."""

    _tags = {
        "object_type": "bootstrap",
    }

    def __init__(self, n_bootstraps=10, block_length=1, random_state=None):
        self.n_bootstraps = n_bootstraps
        self.block_length = block_length
        self.random_state = random_state
        super().__init__()

    def _sample_indices(self, n_obs, size, random_state):
        raise NotImplementedError("abstract method")

    def bootstrap(self, X, test_ratio=0, return_indices=False):
        """Generate bootstrap samples.

        Parameters
        ----------
        X : pd.DataFrame
            Input series represented as a 2D frame.
        test_ratio : float, default=0
            Fraction reserved for test split, ignored beyond controlling sample size.
        return_indices : bool, default=False
            Whether to return sampled integer indices alongside sampled values.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame")

        n_obs = len(X)
        sample_size = int(np.floor(n_obs * (1 - test_ratio)))
        if sample_size <= 0:
            raise ValueError("test_ratio leaves no observations to bootstrap")

        samples = []
        rng = check_random_state(self.random_state)
        for i in range(self.n_bootstraps):
            seed = rng.randint(np.iinfo(np.int32).max)
            idx = self._sample_indices(n_obs=n_obs, size=sample_size, random_state=seed)
            values = X.iloc[idx].to_numpy()
            if return_indices:
                samples.append((values, idx))
            else:
                samples.append(values)

        return samples


class BlockBootstrap(_BaseBootstrap):
    """Block bootstrap with replacement, compatible with TSBootstrapAdapter."""

    def __init__(self, n_bootstraps=10, block_length=1, random_state=None):
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            random_state=random_state,
        )

    def _sample_indices(self, n_obs, size, random_state):
        return _sample_moving_block_indices(
            n_obs=n_obs,
            block_length=self.block_length,
            replacement=True,
            random_state=random_state,
            size=size,
        )


class MovingBlockBootstrap(_BaseBootstrap):
    """Moving block bootstrap without replacement, tsbootstrap-compatible API."""

    def __init__(self, n_bootstraps=10, block_length=1, random_state=None):
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            random_state=random_state,
        )

    def _sample_indices(self, n_obs, size, random_state):
        return _sample_moving_block_indices(
            n_obs=n_obs,
            block_length=self.block_length,
            replacement=False,
            random_state=random_state,
            size=size,
        )
