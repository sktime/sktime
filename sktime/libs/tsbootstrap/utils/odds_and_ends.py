import os
from contextlib import contextmanager
from numbers import Integral

import numpy as np
from numpy.random import Generator


def time_series_split(X: np.ndarray, test_ratio: float):
    """
    Splits a given time series into training and test sets.

    Parameters
    ----------
    X : np.ndarray
        The input time series.
    test_ratio : float
        The ratio of the test set size to the total size of the series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the training set and the test set.
    """
    # Validate test_ratio
    if not 0 <= test_ratio <= 1:
        raise ValueError(
            f"Test ratio must be between 0 and 1. Got {test_ratio}"
        )

    split_index = int(len(X) * (1 - test_ratio))
    return X[:split_index], X[split_index:]


def check_generator(seed_or_rng, seed_allowed: bool = True) -> Generator:
    """Turn seed into a np.random.Generator instance.

    Parameters
    ----------
    seed_or_rng : int, Generator, or None
        If seed_or_rng is None, return the Generator singleton used by np.random.
        If seed_or_rng is an int, return a new Generator instance seeded with seed_or_rng.
        If seed_or_rng is already a Generator instance, return it.
        Otherwise raise ValueError.

    seed_allowed : bool, optional
        If True, seed_or_rng can be an int. If False, seed_or_rng cannot be an int.
        Default is True.

    Returns
    -------
    Generator
        A numpy.random.Generator instance.

    Raises
    ------
    ValueError
        If seed_or_rng is not None, an int, or a numpy.random.Generator instance.
        If seed_or_rng is an int and seed_allowed is False.
        If seed_or_rng is an int and it is not between 0 and 2**32 - 1.
    """
    if seed_or_rng is None:
        return np.random.default_rng()
    if isinstance(seed_or_rng, Generator):
        return seed_or_rng
    if seed_allowed and isinstance(seed_or_rng, Integral):
        if not (0 <= seed_or_rng < 2**32):  # type: ignore
            raise ValueError(
                f"The random seed must be between 0 and 2**32 - 1. Got {seed_or_rng}"
            )
        return np.random.default_rng(seed_or_rng)  # type: ignore

    raise ValueError(
        f"{seed_or_rng} cannot be used to seed a numpy.random.Generator instance"
    )


def generate_random_indices(
    num_samples: Integral, rng: Generator = None
) -> np.ndarray:
    """
    Generate random indices with replacement.

    This function generates random indices from 0 to `num_samples-1` with replacement.
    The generated indices can be used for bootstrap sampling, etc.

    Parameters
    ----------
    num_samples : Integral
        The number of samples for which the indices are to be generated.
        This must be a positive integer.
    rng : Integral, optional
        The seed for the random number generator. If provided, this must be a non-negative integer.
        Default is None, which does not set the numpy's random seed and the results will be non-deterministic.

    Returns
    -------
    np.ndarray
        A numpy array of shape (`num_samples`,) containing randomly generated indices.

    Raises
    ------
    ValueError
        If `num_samples` is not a positive integer or if `random_seed` is provided and
        it is not a non-negative integer.

    Examples
    --------
    >>> generate_random_indices(5, random_seed=0)
    array([4, 0, 3, 3, 3])
    >>> generate_random_indices(5)
    array([2, 1, 4, 2, 0])  # random
    """
    # Check types and values of num_samples and random_seed
    from tsbootstrap.utils.validate import validate_integers

    validate_integers(num_samples, min_value=1)  # type: ignore
    rng = check_generator(rng, seed_allowed=False)

    # Generate random indices with replacement
    in_bootstrap_indices = rng.choice(
        np.arange(num_samples), size=num_samples, replace=True  # type: ignore
    )

    return in_bootstrap_indices


@contextmanager
def suppress_output(verbose: int = 2):
    """A context manager for controlling the suppression of stdout and stderr.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level controlling suppression.
        2 - No suppression (default)
        1 - Suppress stdout only
        0 - Suppress both stdout and stderr

    Returns
    -------
    None

    Examples
    --------
    with suppress_output(verbose=1):
        print('This will not be printed to stdout')
    """
    # No suppression required
    if verbose == 2:
        yield
        return

    # Open null files as needed
    null_fds = [
        os.open(os.devnull, os.O_RDWR) for _ in range(2 if verbose == 0 else 1)
    ]
    # Save the actual stdout (1) and possibly stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)] if verbose == 0 else [os.dup(1)]
    try:
        # Assign the null pointers as required
        os.dup2(null_fds[0], 1)
        if verbose == 0:
            os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back
        for fd, save_fd in zip(null_fds, save_fds):
            os.dup2(save_fd, fd)
        # Close the null files and saved file descriptors
        for fd in null_fds + save_fds:
            os.close(fd)


def _check_nan_inf_locations(
    a: np.ndarray, b: np.ndarray, check_same: bool
) -> bool:
    """
    Check the locations of NaNs and Infs in both arrays.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    check_same : bool
        If True, checks if NaNs and Infs are in the same locations.

    Returns
    -------
    bool
        True if locations do not match and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays have NaNs or Infs in different locations.
    """
    a_nan_locs = np.isnan(a)
    b_nan_locs = np.isnan(b)
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)

    if not np.array_equal(a_nan_locs, b_nan_locs) or not np.array_equal(
        a_inf_locs, b_inf_locs
    ):
        if check_same:
            raise ValueError("NaNs or Infs in different locations")
        else:
            return True

    return False


def _check_inf_signs(a: np.ndarray, b: np.ndarray, check_same: bool) -> bool:
    """
    Check the signs of Infs in both arrays.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    check_same : bool
        If True, checks if Infs have the same signs.

    Returns
    -------
    bool
        True if signs do not match and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays have Infs with different signs.
    """
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)

    if not np.array_equal(np.sign(a[a_inf_locs]), np.sign(b[b_inf_locs])):
        if check_same:
            raise ValueError("Infs with different signs")
        else:
            return True

    return False


def _check_close_values(
    a: np.ndarray, b: np.ndarray, rtol: float, atol: float, check_same: bool
) -> bool:
    """
    Check that the finite values in the arrays are close.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    rtol : float
        The relative tolerance parameter for the np.allclose function.
    atol : float
        The absolute tolerance parameter for the np.allclose function.
    check_same : bool
        If True, checks if the arrays are almost equal.

    Returns
    -------
    bool
        True if values are not close and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays are not almost equal.
    """
    a_nan_locs = np.isnan(a)
    b_nan_locs = np.isnan(b)
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)
    a_masked = np.ma.masked_where(a_nan_locs | a_inf_locs, a)
    b_masked = np.ma.masked_where(b_nan_locs | b_inf_locs, b)

    if check_same:
        if not np.allclose(a_masked, b_masked, rtol=rtol, atol=atol):
            raise ValueError("Arrays are not almost equal")
    else:
        if np.any(~np.isclose(a_masked, b_masked, rtol=rtol, atol=atol)):
            return True

    return False


def assert_arrays_compare(
    a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-8, check_same=True
) -> bool:
    """
    Assert that two arrays are almost equal.

    This function compares two arrays for equality, allowing for NaNs and Infs in the arrays.
    The arrays are considered equal if the following conditions are satisfied:
    1. The locations of NaNs and Infs in both arrays are the same.
    2. The signs of the infinite values in both arrays are the same.
    3. The finite values are almost equal.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    rtol : float, optional
        The relative tolerance parameter for the np.allclose function.
        Default is 1e-5.
    atol : float, optional
        The absolute tolerance parameter for the np.allclose function.
        Default is 1e-8.
    check_same : bool, optional
        If True, raise an AssertionError if the arrays are not almost equal.
        If False, return True if the arrays are not almost equal and False otherwise.
        Default is True.

    Returns
    -------
    bool
        If check_same is False, returns True if the arrays are not almost equal and False otherwise.
        If check_same is True, returns True if the arrays are almost equal and False otherwise.

    Raises
    ------
    AssertionError
        If check_same is True and the arrays are not almost equal.
    ValueError
        If check_same is True and the arrays have NaNs or Infs in different locations.
        If check_same is True and the arrays have Infs with different signs.
    """
    if _check_nan_inf_locations(a, b, check_same):
        return not check_same
    if _check_inf_signs(a, b, check_same):
        return not check_same
    if _check_close_values(a, b, rtol, atol, check_same):
        return not check_same

    return not check_same if not check_same else True
