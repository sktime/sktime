"""Data generators for piecewise linear trends."""

__author__ = ["Tveten"]

import numbers

import numpy as np
import pandas as pd
import scipy.stats

from ..utils.validation.generation import check_random_generator, check_segment_lengths
from ._utils import recycle_list


def generate_continuous_piecewise_linear_data(
    slopes: float | list[float] | None = None,
    lengths: int | list[int] | np.ndarray | None = None,
    *,
    n_segments: int = 3,
    n_samples: int = 100,
    intercept: float = 0.0,
    noise_std: float = 1.0,
    seed: int | None = None,
    return_params: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate a continuous piecewise linear signal with noise.

    Parameters
    ----------
    slopes : float, list of floats, optional (default=None)
        Slopes for each segment.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If None, slopes alternate between 1.0 and -1.0.

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

    intercept : float, default=0
        Starting intercept value.

    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise to add.

    seed : np.random.Generator | int | None, optional
        Seed for the random number generator or a numpy random generator instance.
        If specified, this ensures reproducible output across multiple calls.

    return_params : bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column containing the generated data.

    dict, optional
        If `return_params` is True, a dictionary containing the parameters used to
        generate the data, including:

        - `n_segments`: Number of segments.
        - `n_samples`: Total number of samples.
        - `lengths`: Segment lengths.
        - `slopes`: Slopes for each segment.
        - `intercept`: Intercept value.
        - `noise_std`: Standard deviation of the noise.
        - `change_points`: Indices where the slope changes.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative. Got {noise_std}.")

    random_generator = check_random_generator(seed)
    lengths = check_segment_lengths(
        lengths, n_segments, n_samples, seed=random_generator
    )
    n_segments = len(lengths)
    n_samples = np.sum(lengths)

    if slopes is None:
        slopes = [1.0, -1.0]
    if isinstance(slopes, numbers.Number):
        slopes = [slopes]
    slopes = recycle_list(slopes, n_segments)

    time = np.arange(n_samples)
    signal = np.zeros(n_samples)
    change_points = np.cumsum(lengths)[:-1]  # The last point is the end of the data.

    # First segment
    signal[: change_points[0]] = intercept + slopes[0] * time[: change_points[0]]
    current_value = signal[change_points[0] - 1]

    # Middle segments
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        segment_time = time[start_idx:end_idx] - time[start_idx]
        signal[start_idx:end_idx] = current_value + slopes[i + 1] * segment_time
        current_value = signal[end_idx - 1]

    # Last segment
    if len(change_points) > 0:
        last_start = change_points[-1]
        segment_time = time[last_start:] - time[last_start]
        signal[last_start:] = current_value + slopes[-1] * segment_time

    signal += scipy.stats.norm.rvs(
        loc=0, scale=noise_std, size=n_samples, random_state=random_generator
    )
    generated_df = pd.DataFrame(signal)
    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": n_samples,
            "lengths": lengths,
            "slopes": slopes,
            "intercept": intercept,
            "noise_std": noise_std,
            "change_points": change_points,
        }
        return generated_df, params

    return generated_df


def generate_continuous_piecewise_linear_signal(
    change_points, slopes, intercept=0, n_samples=200, noise_std=0.1, random_seed=None
):
    """Generate a continuous piecewise linear signal with noise.

    Parameters
    ----------
    change_points : list
        List of indices where the slope changes (kink points)
    slopes : list
        List of slopes for each segment (should be one more than change_points)
    intercept : float, default=0
        Starting intercept value
    n_samples : int, default=200
        Total number of samples
    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise to add
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with the signal and corresponding time points
    list
        List of true change points (as indices)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(slopes) != len(change_points) + 1:
        raise ValueError(  # pragma: no cover
            "Number of slopes must be one more than number of change points"
        )

    # Create time points and allocate signal
    time = np.arange(n_samples)
    signal = np.zeros(n_samples)

    # First segment
    signal[: change_points[0]] = intercept + slopes[0] * time[: change_points[0]]
    current_value = signal[change_points[0] - 1]

    # Middle segments
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        segment_time = time[start_idx:end_idx] - time[start_idx]
        signal[start_idx:end_idx] = current_value + slopes[i + 1] * segment_time
        current_value = signal[end_idx - 1]

    # Last segment
    if len(change_points) > 0:
        last_start = change_points[-1]
        segment_time = time[last_start:] - time[last_start]
        signal[last_start:] = current_value + slopes[-1] * segment_time

    # Add noise
    signal += np.random.normal(0, noise_std, n_samples)

    # Convert to DataFrame
    df = pd.DataFrame({"signal": signal})

    return df
