"""Data generators for piecewise normal data."""

__author__ = ["Tveten"]

from numbers import Number

import numpy as np
import pandas as pd
import scipy.stats

from ..utils.validation.generation import check_random_generator, check_segment_lengths
from ._generate import generate_piecewise_data
from ._utils import recycle_list


def get_n_variables(
    n_variables: int,
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
) -> int:
    """Derive the number of variables from the input parameters."""
    # Convert to list if not, to make the rest of the code easier.
    if not isinstance(means, list):
        means = [means]
    if not isinstance(variances, list):
        variances = [variances]

    mean_n_vars = max(
        len(mean) if isinstance(mean, (list, np.ndarray)) else 1 for mean in means
    )
    var_n_vars = max(
        len(var) if isinstance(var, (list, np.ndarray)) else 1 for var in variances
    )
    return int(max(n_variables, mean_n_vars, var_n_vars))


def _check_affected_variables(
    proportion_affected: float | list[float] | np.ndarray | None,
    randomise_affected_variables: bool,
    n_segments: int,
    n_variables: int,
    random_state: np.random.Generator,
) -> list[np.ndarray]:
    """Check and return affected variables for each segment."""
    if proportion_affected is None:
        proportion_affected = scipy.stats.uniform(1e-8, 1).rvs(
            size=n_segments, random_state=random_state
        )
    elif isinstance(proportion_affected, Number):
        proportion_affected = [proportion_affected]

    if len(proportion_affected) == 0:
        raise ValueError("proportion_affected cannot be an empty list or np.array.")

    proportion_affected = recycle_list(list(proportion_affected), n_segments)

    for prop in proportion_affected:
        if not (0 < prop <= 1):
            raise ValueError(
                "Proportion of affected variables must be between (0, 1]."
                f" Got `proportion_affected`={prop}."
            )

    affected_variables = []
    for prop in proportion_affected:
        affected_size = int(np.ceil(n_variables * prop))
        if randomise_affected_variables:
            affected_vars = np.sort(
                random_state.choice(n_variables, size=affected_size, replace=False)
            )
        else:
            affected_vars = np.arange(affected_size)
        affected_variables.append(affected_vars)

    return affected_variables


def _check_means(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Check and return means for each segment."""
    if means is None or isinstance(means, (Number, np.ndarray)):
        means = [means] * n_segments

    n_variables = int(n_variables)
    _means = [np.zeros(n_variables)]  # Initialize for the loop to work. Remove later.
    for mean, affected in zip(means, affected_variables):
        prev_mean = _means[-1].copy()
        if mean is None:
            # The affected set are the variables that change, so the next mean vector
            # should be the same, except for the affected variables.
            _mean = prev_mean
            _mean[affected] = scipy.stats.norm(0, 2).rvs(
                size=affected.size, random_state=random_state
            )
        elif isinstance(mean, Number):
            _mean = prev_mean
            _mean[affected] = mean
        else:
            _mean = np.asarray(mean).reshape(-1)

        if _mean.shape[0] != int(n_variables):
            raise ValueError(
                "Mean vector must have the same length as the number of variables."
                f" Got mean={_mean} with shape {_mean.shape}"
                f" and n_variables={n_variables}."
            )
        _means.append(_mean)

    _means = _means[1:]  # The first element is just to initialize the loop.

    # Means are recycled to match the number of segments. Will only have an effect in
    # cases where `means` is a list and the list is shorter than `n_segments`.
    _means = recycle_list(_means, n_segments)
    return _means


def _check_variances(
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Create covariance matrices for each segment."""
    if variances is None or isinstance(variances, (Number, np.ndarray)):
        variances = [variances] * n_segments

    n_variables = int(n_variables)
    _vars = [np.ones(n_variables)]  # Initialize for the loop to work.
    _covs = []
    for cov, affected in zip(variances, affected_variables):
        prev_var = _vars[-1].copy()
        if cov is None:
            # The affected set are the variables that change, so the next covariance
            # matrix should be the same, except for the affected variables.
            _var = prev_var
            _var[affected] = scipy.stats.chi2(2).rvs(
                size=affected.size, random_state=random_state
            )
        elif isinstance(cov, Number):
            _var = prev_var
            _var[affected] = cov
        else:
            _var = np.asarray(cov)

        if _var.ndim == 1:
            _cov = np.diag(_var)
        elif _var.ndim == 2 and _var.shape[0] == _var.shape[1]:
            _cov = _var
        else:
            raise ValueError(
                "Covariance matrix must be a square matrix with shape (p, p)."
                f" Got covariance matrix with shape {_var.shape} and p={n_variables}."
            )

        if not np.allclose(_cov, _cov.T, atol=1e-8):
            raise ValueError("Covariance matrix must be symmetric.")

        eigvals = np.linalg.eigvalsh(_cov)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix must be positive definite.")

        _covs.append(_cov)

    # Covariance matrices are recycled to match the number of segments. Will only have
    # an effect in cases where `variances` is a list and the list is shorter than
    # `n_segments`.
    _covs = recycle_list(_covs, n_segments)
    return _covs


def generate_piecewise_normal_data(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = 1.0,
    lengths: int | list[int] | np.ndarray | None = None,
    *,
    n_segments: int = 3,
    n_samples: int = 100,
    n_variables: int = 1,
    proportion_affected: float | list[float] | np.ndarray | None = None,
    randomise_affected_variables: bool = False,
    seed: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> pd.DataFrame:
    """Generate piecewise multivariate normal data.

    Generates piecewise multivariate normal data, where the distributional changes
    from one segment to another may be sparse. E.g., the difference between two
    mean vectors may only have a few non-zero elements.

    Parameters
    ----------
    means : float, list of float, or list of np.ndarray, optional (default=None)
        Means for each segment.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If floats, they are used for all affected variables (see `proportion_affected`)
        If None, random means are generated from a normal distribution with mean 0
        and standard deviation 2.

    variances : float, list of float, or list of np.ndarray, optional (default=1.0)
        Variances or covariance matrices for each segment. Vectors are treated as
        diagonal covariance matrices.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If floats, they are used for all affected variables (see `proportion_affected`)
        If None, random variances are generated from a chi-squared distribution with
        2 degrees of freedom.

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

    n_variables : int, optional (default=1)
        Number of variables (columns) in the generated data.

    proportion_affected: float, list of float, or np.ndarray, optional (default=None)
        Proportion of variables affected by each change.
        I.e., the proportion of non-zero elements in the differences between adjacent
        means or variances.
        Only applies when `means` and `variances` are None or floats and
        `n_variables > 1`.
        All proportions must be in (0, 1].
        The number of affected variables is determined as
        `int(np.ceil(n_variables * proportion_affected))`.
        The proportions are recycled to match the number of segments specified by
        `lengths` or `n_segments`.
        If None, a random proportion of variables is affected.

    randomise_affected_variables : bool, optional (default=False)
        If True, the affected variables are randomly selected for each change point.
        If False, the first variables are affected.

    seed : np.random.Generator | int | None, optional
        Seed for the random number generator or a numpy random generator instance.
        If specified, this ensures reproducible output across multiple calls.

    return_params : bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with generated data.

    dict
        Dictionary containing the parameters used to generate the data.
        Keys: `"n_segments"`, `"n_samples"`, `"means"`, `"variances"`,
        `"lengths"`, `"change_points"` (the start indices of each segment), and
        `"affected_variables"` (which variables among 0:n_variables are affected by
        each change).
        Returned only if `return_params` is True.

    Examples
    --------
    >>> # Example 1: Two segments with specified means
    >>> from skchange.datasets import generate_piecewise_normal_data
    >>> df = generate_piecewise_normal_data(
    ...     means=[0, 5], lengths=5, n_segments=2, n_variables=1, seed=0
    ... )
    >>> df
              0
    0  0.640423
    1  0.104900
    2 -0.535669
    3  0.361595
    4  1.304000
    5  5.947081
    6  4.296265
    7  3.734579
    8  4.376726
    9  5.041326

    >>> # Example 2: Unspecified means, variances and lengths
    >>> df, params = generate_piecewise_normal_data(
    ...     n_samples=10, n_segments=2, n_variables=2, seed=1, return_params=True
    ... )
    >>> df
              0         1
    0 -3.143268  2.391830
    1 -2.241742  2.104844
    2 -2.577892  2.357425
    3 -3.342769  1.647802
    4 -3.088434  2.409558
    5  0.932471  1.518255
    6  0.110841  1.553519
    7  0.900891  1.535109
    8  2.186813  2.817436
    9 -1.818413 -0.078302
    >>> params
    {'n_segments': 2,
    'n_samples': np.int64(10),
    'means': [array([-2.60631446,  1.81071173]), array([0.89274914, 1.81071173])],
    'variances': [array([[1., 0.],
            [0., 1.]]),
    array([[1., 0.],
            [0., 1.]])],
    'lengths': array([5, 5]),
    'change_points': array([5]),
    'affected_variables': [array([0, 1]), array([0])]}
    """
    random_generator = check_random_generator(seed)
    if n_variables < 1:
        raise ValueError("n_variables must be at least 1.")

    lengths = check_segment_lengths(
        lengths, n_segments, n_samples, seed=random_generator
    )
    n_segments = len(lengths)

    _n_variables = get_n_variables(n_variables, means, variances)
    affected_variables = _check_affected_variables(
        proportion_affected,
        randomise_affected_variables,
        n_segments,
        _n_variables,
        random_generator,
    )
    means = _check_means(
        means,
        n_segments,
        _n_variables,
        affected_variables,
        random_generator,
    )
    covs = _check_variances(
        variances,
        n_segments,
        _n_variables,
        affected_variables,
        random_generator,
    )
    distributions = [
        scipy.stats.multivariate_normal(mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ]
    df, _params = generate_piecewise_data(
        distributions=distributions,
        lengths=lengths,
        seed=random_generator,
        return_params=True,
    )
    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": _params["n_samples"],
            "means": means,
            "variances": covs,
            "lengths": _params["lengths"],
            "change_points": _params["change_points"],
            "affected_variables": affected_variables,
        }
        return df, params
    else:
        return df


def _check_change_points(change_points: int | list[int], n: int) -> list[int]:
    """Check if change points are valid.

    Parameters
    ----------
    change_points : list of int
        List of change points.
    n : int
        Total number of observations.

    Raises
    ------
    ValueError
        If any change point is out of bounds.
    """
    if isinstance(change_points, Number):
        change_points = [change_points]

    change_points = sorted(change_points)
    if any([cpt > n - 1 or cpt < 0 for cpt in change_points]):
        raise ValueError(
            "Change points must be within the range of the data."
            f" Got n={n}, max(change_points)={change_points} and"
            f" min(change_points)={min(change_points)}."
        )
    if len(change_points) >= 2 and min(np.diff(change_points)) < 1:
        raise ValueError(
            "Change points must be at least 1 apart."
            f" Got change_points={change_points}."
        )

    return change_points


def generate_changing_data(
    n: int = 100,
    changepoints: int | list[int] = 50,
    means: float | list[float] | list[np.ndarray] = 0.0,
    variances: float | list[float] | list[np.ndarray] = 1.0,
    random_state: int = None,
):
    """
    Generate piecewise multivariate normal data with changing means and variances.

    DEPRECATED: Use `generate_piecewise_normal_data` instead.

    Parameters
    ----------
    n : int, optional, default=100
        Number of observations.
    changepoints : int or list of ints, optional, default=50
        Changepoints in the data.
    means : list of floats or list of arrays, optional, default=0.0
        List of means for each segment.
    variances : list of floats or list of arrays, optional, default=1.0
        List of variances for each segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    change_points = _check_change_points(changepoints, n)

    if isinstance(means, Number):
        means = [means]
    if isinstance(variances, Number):
        variances = [variances]

    means = [np.asarray(mean).reshape(-1) for mean in means]
    variances = [np.asarray(variance).reshape(-1) for variance in variances]

    n_segments = len(change_points) + 1
    if len(means) == 1:
        means = means * n_segments
    if len(variances) == 1:
        variances = variances * n_segments

    if n_segments != len(means) or n_segments != len(variances):
        raise ValueError(
            "Number of segments (len(changepoints) + 1),"
            + " means and variances must be the same."
        )

    p = len(means[0])
    x = scipy.stats.multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    change_points = [0] + change_points + [n]
    for prev_cpt, next_cpt, mean, variance in zip(
        change_points[:-1], change_points[1:], means, variances
    ):
        x[prev_cpt:next_cpt] = mean + np.sqrt(variance) * x[prev_cpt:next_cpt]

    out_columns = [f"var{i}" for i in range(p)]
    df = pd.DataFrame(x, index=range(len(x)), columns=out_columns)
    return df


def generate_anomalous_data(
    n: int = 100,
    anomalies: tuple[int, int] | list[tuple[int, int]] = (70, 80),
    means: float | list[float] | list[np.ndarray] = 3.0,
    variances: float | list[float] | list[np.ndarray] = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate multivariate normal data with anomalies.

    DEPRECATED: Use `generate_piecewise_normal_data` instead.

    Parameters
    ----------
    n : int, optional (default=100)
        Number of observations.
    anomalies : list of tuples, optional (default=[(71, 80)])
        List of tuples of the form [start, end) indicating the start and end of an
        anomaly.
    means : list of floats or list of arrays, optional (default=[0.0])
        List of means for each segment.
    variances : list of floats or list of arrays, optional (default=[1.0])
        List of variances for each segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    if isinstance(anomalies, tuple):
        anomalies = [anomalies]
    if isinstance(means, Number):
        means = [means]
    if isinstance(variances, Number):
        variances = [variances]

    means = [np.asarray(mean).reshape(-1) for mean in means]
    variances = [np.asarray(variance).reshape(-1) for variance in variances]

    if len(means) == 1:
        means = means * len(anomalies)
    if len(variances) == 1:
        variances = variances * len(anomalies)

    if len(anomalies) != len(means) or len(anomalies) != len(variances):
        raise ValueError("Number of anomalies, means and variances must be the same.")
    if any([len(anomaly) != 2 for anomaly in anomalies]):
        raise ValueError("Anomalies must be of length 2.")
    if any([anomaly[1] <= anomaly[0] for anomaly in anomalies]):
        raise ValueError("The start of an anomaly must be before its end.")
    if any([anomaly[0] < 0 for anomaly in anomalies]):
        raise ValueError("Anomalies must start at a non-negative index.")
    if any([anomaly[1] > n for anomaly in anomalies]):
        raise ValueError("Anomalies must be within the range of the data.")

    p = len(means[0])
    x = scipy.stats.multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    for anomaly, mean, variance in zip(anomalies, means, variances):
        start, end = anomaly
        x[start:end] = mean + np.sqrt(variance) * x[start:end]

    out_columns = [f"var{i}" for i in range(p)]
    df = pd.DataFrame(x, index=range(len(x)), columns=out_columns)
    return df


def generate_alternating_data(
    n_segments: int,
    segment_length: int,
    p: int = 1,
    mean: float = 0.0,
    variance: float = 1.0,
    affected_proportion: float = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate multivariate normal data that is alternating between two states.

    DEPRECATED: Use `generate_piecewise_normal_data` instead.

    The data alternates between a state with mean 0 and variance 1 and a state with
    mean `mean` and variance `variance`. The length of the segments are all identical
    and equal to `segment_length`. The proportion of components that are affected by
    the change is determined by `affected_proportion`.

    Parameters
    ----------
    n_segments : int
        Number of segments to generate.
    segment_length : int
        Length of each segment.
    p : int, optional (default=1)
        Number of dimensions.
    mean : float, optional (default=0.0)
        Mean of every other segment.
    variance : float, optional (default=1.0)
        Variances of every other segment.
    affected_proportion : float, optional (default=1.0)
        Proportion of components {1, ..., p} that are affected by each change in
        every other segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    means = []
    vars = []
    n_affected = int(np.round(p * affected_proportion))
    for i in range(n_segments):
        zero_mean = [0] * p
        changed_mean = [mean] * n_affected + [0] * (p - n_affected)
        mean_vec = zero_mean if i % 2 == 0 else changed_mean
        means.append(mean_vec)
        one_var = [1] * p
        changed_var = [variance] * n_affected + [1] * (p - n_affected)
        vars_vec = one_var if i % 2 == 0 else changed_var
        vars.append(vars_vec)

    n = segment_length * n_segments
    changepoints = [segment_length * i for i in range(1, n_segments)]
    return generate_changing_data(n, changepoints, means, vars, random_state)
