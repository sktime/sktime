"""Utility functions for validating data generation in skchange."""

import numbers

import numpy as np


def check_random_generator(
    seed: np.random.Generator | int | None,
) -> np.random.Generator:
    """Check and turn a seed into a numpy random generator.

    Parameters
    ----------
    seed : int or np.random.Generator or None
        If None, return a new np.random.default_rng() instance.
        If an int, return a new np.random.default_rng(seed).
        If a np.random.Generator, return it as is.

    Returns
    -------
    np.random.Generator
        Random generator instance based on the provided seed.
    """
    if seed is None:
        return np.random.default_rng()
    elif isinstance(seed, int):
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        return seed
    else:
        raise TypeError(
            "random_state must be a numpy.random.Generator instance, an int or None."
            f" Got {type(seed)}."
        )


def check_segment_lengths(
    lengths: int | list[int] | np.ndarray | None,
    n_segments: int,
    n_samples: int,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Check and return segment lengths.

    Parameters
    ----------
    lengths : int, list of int or np.ndarray
        If None, generate `n_segments` random segment lengths with a total sample size
        of `n_samples`.
        If an int, generate `n_segments` segment lengths of length `lengths`.
        If a list or numpy array, ensure it is 1D and convert to numpy array.

    n_samples : int
        Total number of samples to generate.

    n_segments : int
        Number of segments to generate.

    seed : int | np.random.Generator | None, optional
        Seed for the random number generator. Used to generate random segment lengths
        if `lengths` is None.

    Returns
    -------
    np.ndarray
        1d array of segment lengths.
    """
    random_generator = check_random_generator(seed)

    if n_samples < 1 or not isinstance(n_samples, numbers.Integral):
        raise ValueError(f"n_samples must be a positive integer. Got {n_samples}.")
    if n_segments < 1 or not isinstance(n_segments, numbers.Integral):
        raise ValueError(f"n_segments must be a positive integer. Got {n_segments}.")
    if n_samples < n_segments:
        raise ValueError(
            f"n_samples ({n_samples}) must be at least n_segments ({n_segments})."
        )

    if lengths is None:
        if n_segments == 1:
            lengths = [n_samples]
        else:
            change_points = random_generator.choice(
                np.arange(1, n_samples), size=n_segments - 1, replace=False
            )
            lengths = np.diff(
                np.concatenate(([0], np.sort(change_points), [n_samples]))
            )
    elif isinstance(lengths, numbers.Integral):
        lengths = [lengths] * n_segments
    elif isinstance(lengths, (list, np.ndarray)):
        if len(lengths) == 0:
            raise ValueError("lengths must not be an empty list or array.")
        lengths = np.asarray(lengths, dtype=int)
        if lengths.ndim != 1:
            raise ValueError("lengths must be a 1d array.")
    else:
        raise TypeError(
            "lengths must be an integer, a list of integers, or a 1d numpy array."
            f" Got {type(lengths)}."
        )

    lengths = np.asarray(lengths, dtype=int)
    if np.any(lengths <= 0):
        raise ValueError(
            f"All lengths must be positive integers."
            f" Found entries{lengths[lengths <= 0]}."
        )

    return lengths
