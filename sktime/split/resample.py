"""Single-window resamplers and composition with a forecasting horizon."""

import numpy as np

from sktime.split.base import BaseSplitter
from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh


class SlidingWindowResampler(BaseSplitter):
    """Generate individual fixed-length windows, without a test fold.

    Each iteration of ``split`` returns one integer array, ``split_loc`` returns
    one index, and ``split_series`` returns one data container. Full windows
    start at position zero and include the last observation when the step allows.
    Window sizes are measured in observations, including for a datetime index.
    Panel and hierarchical inputs are resampled independently per series.

    Parameters
    ----------
    window_length : int, default=10
        Positive number of observations in a window.
    step_length : int, default=1
        Positive number of observations between successive window ends.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import SlidingWindowResampler
    >>> cv = SlidingWindowResampler(window_length=3, step_length=2)
    >>> [w.tolist() for w in cv.split(np.arange(7))]
    [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    """

    _tags = {"split:arity": 1}

    def __init__(self, window_length=10, step_length=1):
        self.window_length = window_length
        self.step_length = step_length
        super().__init__()

    def __post_init__(self):
        """Validate window parameters after construction."""
        for name in ["window_length", "step_length"]:
            value = getattr(self, name)
            if not is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        super().__post_init__()

    def _fh(self):
        """Return None because resampling has no forecasting horizon."""
        return None

    def _split(self, y):
        for end in range(self.window_length, len(y) + 1, self.step_length):
            yield np.arange(end - self.window_length, end, dtype=int)

    def get_cutoffs(self, y=None):
        """Return the positional end of each resampled window."""
        if y is None:
            raise ValueError("y is required to determine window cutoffs.")
        return np.asarray([window[-1] for window in self.split(y)], dtype=int)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return [{"window_length": 3}, {"window_length": 5, "step_length": 2}]


class ExpandingWindowResampler(SlidingWindowResampler):
    """Generate individual expanding windows, without a test fold.

    Parameters
    ----------
    window_length : int, default=10
        Positive number of observations in the first window.
    step_length : int, default=1
        Positive number of observations added to each successive window.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ExpandingWindowResampler
    >>> cv = ExpandingWindowResampler(window_length=3, step_length=2)
    >>> [w.tolist() for w in cv.split(np.arange(7))]
    [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]]
    """

    def _split(self, y):
        for end in range(self.window_length, len(y) + 1, self.step_length):
            yield np.arange(end, dtype=int)


class ResamplerToSplitter(BaseSplitter):
    """Add a relative forecasting horizon to a single-window resampler.

    Each resampled window becomes a training fold. The test fold contains
    observations at ``fh`` positions after its last observation. Windows whose
    full test fold extends beyond the series are omitted. For panel input this
    composition is performed per series before broadcasting the resulting folds.

    Parameters
    ----------
    resampler : BaseSplitter with tag ``split:arity=1``
        Resampler that generates one non-empty, increasing window per iteration.
    fh : int or array-like of int, default=1
        Positive relative forecast steps. An integer denotes one forecast step.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ResamplerToSplitter, SlidingWindowResampler
    >>> cv = ResamplerToSplitter(SlidingWindowResampler(3, 2), fh=[1, 2])
    >>> [(a.tolist(), b.tolist()) for a, b in cv.split(np.arange(7))]
    [([0, 1, 2], [3, 4]), ([2, 3, 4], [5, 6])]
    """

    def __init__(self, resampler, fh=1):
        self.resampler = resampler
        self.fh = fh
        super().__init__()

    def __post_init__(self):
        """Validate composition without changing the component resampler."""
        if (
            not isinstance(self.resampler, BaseSplitter)
            or self.resampler.get_tag("split:arity", 2) != 1
        ):
            raise ValueError("resampler must be a BaseSplitter with split:arity=1.")
        steps = check_fh(self.fh, enforce_relative=True).to_numpy()
        if not np.issubdtype(steps.dtype, np.integer) or np.any(steps <= 0):
            raise ValueError("fh must contain positive integer steps.")
        super().__post_init__()

    def _split(self, y):
        steps = check_fh(self.fh, enforce_relative=True).to_numpy()
        for window in self.resampler.split(y):
            if len(window) == 0:
                continue
            if np.any(np.diff(window) <= 0):
                raise ValueError("Resampled windows must be strictly increasing.")
            test = window[-1] + steps
            if test[-1] < len(y):
                yield window, test

    def get_cutoffs(self, y=None):
        """Return positional cutoffs of folds with a complete test horizon."""
        if y is None:
            raise ValueError("y is required to determine split cutoffs.")
        return np.asarray([train[-1] for train, _ in self.split(y)], dtype=int)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return [
            {"resampler": SlidingWindowResampler(3), "fh": [1, 2]},
            {"resampler": ExpandingWindowResampler(5, 2)},
        ]
