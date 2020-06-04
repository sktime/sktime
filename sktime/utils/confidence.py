from scipy.stats import norm

__all__ = ("zscore",)


def zscore(level: float, two_tailed: bool = True) -> float:
    """
    Calculate a z-score from a confidence level.

    Parameters
    ----------

    level : float
        A confidence level, in the open interval (0, 1).

    two_tailed : bool (default=True)
        If True, return the two-tailed z score.

    Returns
    -------

    z : float
        The z score.
    """
    alpha = 1 - level
    if two_tailed:
        alpha /= 2

    return -norm.ppf(alpha)
