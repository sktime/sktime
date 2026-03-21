"""Data generators."""

__author__ = ["Tveten"]

import numpy as np
import pandas as pd
import scipy.stats

from ..utils.validation.generation import check_random_generator, check_segment_lengths
from ._utils import recycle_list


def _check_distributions(
    distributions: (
        scipy.stats.rv_continuous
        | scipy.stats.rv_discrete
        | list[scipy.stats.rv_continuous]
        | list[scipy.stats.rv_discrete]
        | None
    ),
    n_segments: int,
) -> tuple[list[scipy.stats.rv_continuous | scipy.stats.rv_discrete], int, np.dtype]:
    """Check if distributions are valid and return as a list.

    Parameters
    ----------
    distributions : list of `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`
        List of distributions for each segment.

    n_segments : int
        Number of segments to generate. Used to check if the number of distributions
        matches the number of segments.

    Returns
    -------
    list[scipy.stats.rv_continuous | scipy.stats.rv_discrete]
        List of distributions for each segment, where each distribution is guarnteed
        to have a `rvs(size: int, random_state: int | None)` method that returns
        a numpy array or scalar of the same size, and the output size is either
        1 or `p`.
    int
        Output size of the distributions, which is either 1 or `p`.
    np.dtype
        Data type of the output of the distributions.
    """
    if distributions is None:
        distributions = [scipy.stats.norm(), scipy.stats.norm(5)]
    elif not isinstance(distributions, list):
        distributions = [distributions]

    if len(distributions) == 0:
        raise ValueError("distributions cannot be an empty list.")

    distributions = recycle_list(distributions, n_segments)

    output_sizes = []
    output_dtypes = []
    for dist in distributions:
        try:
            output = dist.rvs(size=1)
            output_sizes.append(output.size)
            output_dtypes.append(output.dtype)
        except Exception:
            output_sizes.append(None)

    if any(size is None for size in output_sizes):
        raise ValueError(
            "All distributions must support the 'rvs' method with a 'size' argument,"
            " where the output is a numpy.array or numpy scalar."
            " Ensure that all distributions are valid scipy.stats distributions."
        )

    if len(set(output_sizes)) > 1:
        raise ValueError(
            f"All distributions must produce samples with the same number of variables."
            f" Got distribution.rvs(size=1).size outputs: {output_sizes}."
        )

    if len(set(output_dtypes)) > 1:
        raise ValueError(
            "All distributions must produce samples with the same data type."
            f" Got distribution.rvs(size=1).dtype outputs: {output_dtypes}."
        )

    return distributions, output_sizes[0], output_dtypes[0]


def generate_piecewise_data(
    distributions: scipy.stats.rv_continuous
    | scipy.stats.rv_discrete
    | list[scipy.stats.rv_continuous]
    | list[scipy.stats.rv_discrete]
    | None = None,
    lengths: int | list[int] | np.ndarray | None = None,
    *,
    n_segments: int = 3,
    n_samples: int = 100,
    seed: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate data with a piecewise constant distribution.

    Generate piecewise segments of data from `scipy.stats` distributions, where
    unspecified parameters are randomly generated.

    Parameters
    ----------
    distributions : list of `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`, optional (default=None)
        The distributions for generating piecewise data.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If None, alternating segments of `scipy.stats.norm()` and `scipy.stats.norm(5)`
        are used. Each distribution is expected to be a scipy distribution instance
        (e.g., `scipy.stats.norm`, `scipy.stats.uniform`). See
        `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
        for a list of all available distributions.

    lengths : int, list of int or np.ndarray, optional (default=None)
        The segment lengths. There are three possible cases:

        1. `list` or `numpy array`: Custom set of segment lengths.
        2. `int`: Length of `n_segments` equal segments.
        3. `None`: Generate `n_segments` random segment lengths with a total sample size
           of `n_samples`.

    n_segments : int (default=3)
        Number of segments to generate if `lengths` is an integer or None.

    n_samples : int (default=100)
        Total number of samples to generate if `lengths` is not specified.

    seed : np.random.Generator | int | None, optional
        Seed for the random number generator or a numpy random generator instance.
        If specified, this ensures reproducible output across multiple calls.

    return_params : bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data.

    Returns
    -------
    pd.DataFrame
        Data frame with generated data.

    dict, optional
        A dictionary containing the parameters used to generate the data. Only returned
        if `return_params` is True. It has the following keys:

        * `"n_segments"` : number of segments generated.
        * `"n_samples"` : total number of samples generated.
        * `"distributions"` : list of `scipy.stats.rv_continuous` or
          `scipy.stats.rv_discrete` with the distributions used for each segment.
        * `"lengths"` : list of lengths for each segment.
        * `"change_points"` : list of change points, which are the starting indices
          of each segment in the data.

    Examples
    --------
    >>> # Example 1: Two normal segments
    >>> from skchange.datasets import generate_piecewise_data
    >>> from scipy.stats import norm
    >>> generate_piecewise_data(
    ...     distributions=[norm(0, 1), norm(10, 0.1)],
    ...     lengths=[7, 3],
    ...     seed=1,
    ... )
               0
    0   0.345584
    1   0.821618
    2   0.330437
    3  -1.303157
    4   0.905356
    5   0.446375
    6  -0.536953
    7  10.058112
    8  10.036457
    9  10.029413

    >>> # Example 2: Two Poisson segments
    >>> from scipy.stats import poisson
    >>> generate_piecewise_data(
    ...     distributions=[poisson(1), poisson(10)],
    ...     lengths=[5, 5],
    ...     seed=2,
    ... )
        0
    0   0
    1   0
    2   1
    3   2
    4   0
    5   8
    6  11
    7   9
    8   9
    9   9


    >>> # Example 3: Specify int lengths and n_segments
    >>> generate_piecewise_data(
    ...     distributions=[norm(0), norm(5)],
    ...     lengths=3,
    ...     n_segments=3,
    ...     seed=3,
    ... )
              0
    0  2.040919
    1 -2.555665
    2  0.418099
    3  4.432230
    4  4.547351
    5  4.784403
    6 -2.019986
    7 -0.231932
    8 -0.865213
    """  # noqa: E501
    random_generator = check_random_generator(seed)
    lengths = check_segment_lengths(
        lengths, n_segments, n_samples, seed=random_generator
    )
    n_segments = len(lengths)
    n_samples = np.sum(lengths)
    distributions, n_variables, dtype = _check_distributions(distributions, n_segments)

    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    generated_values = np.empty((n_samples, n_variables), dtype=dtype)
    for distribution, start, end in zip(distributions, starts, ends):
        length = end - start
        values = distribution.rvs(size=length, random_state=random_generator)
        generated_values[start:end, :] = values.reshape(length, n_variables)

    generated_df = pd.DataFrame(generated_values)

    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": n_samples,
            "lengths": lengths,
            "distributions": distributions,
            "change_points": starts[1:],
        }
        return generated_df, params

    return generated_df
