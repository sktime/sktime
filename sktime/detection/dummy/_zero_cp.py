"""Dummy change point detector which detects no change points."""

from sktime.detection.dummy._zero_an import ZeroAnomalies


class ZeroChangePoints(ZeroAnomalies):
    """Dummy change point detector which detects no change points ever.

    Naive method that can serve as benchmarking pipeline or API test.

    Detects no change points.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import ZeroChangePoints
    >>> X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d = ZeroChangePoints()
    >>> Xt = d.fit_transform(X)
    """

    # same code except the tag
    _tags = {
        "task": "change_point_detection",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }
