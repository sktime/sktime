"""Dummy change point detector which detects change points after steps."""

from sktime.detection.dummy._dummy_regular_an import DummyRegularAnomalies


class DummyRegularChangePoints(DummyRegularAnomalies):
    """Dummy change point detector which detects a change point every x steps.

    Naive method that can serve as benchmarking pipeline or API test.

    Detects a change point every ``step_size`` location indices.
    The first change point is detected at location index ``step_size``,
    the second at ``2 * step_size``, and so on.

    Parameters
    ----------
    step_size : int, default=2
        The step size at which change points are detected.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyRegularChangePoints
    >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d = DummyRegularChangePoints(step_size=3)
    >>> yt = d.fit_transform(y)
    """

    # same code except the tag
    _tags = {"task": "change_point_detection"}
