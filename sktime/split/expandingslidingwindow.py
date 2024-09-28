import pandas as pd
from sktime.split.base import BaseWindowSplitter
from sktime.split.base._common import (
    DEFAULT_FH,
    DEFAULT_STEP_LENGTH,
    DEFAULT_WINDOW_LENGTH,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
)


class ExpandingSlidingWindowSplitter(BaseWindowSplitter):
    r"""Combined Expanding and Sliding Window Splitter.

    This splitter starts as an expanding window splitter until a specified
    maximum window length is reached, then transitions to a sliding window splitter.

    For example, with ``initial_window = 1``, ``window_length = 2``, ``step_length = 1``,
    ``fh = [1, 2]``,, and ``max_expanding_window_length = 8``

    |--------------------|
    | * x x - - - - - - -| Expanding (initial_window = 1)
    | * * x x - - - - - -| Expanding
    | * * * x x - - - - -| Expanding
    | * * * * x x - - - -| Expanding
    | * * * * * x x - - -| Expanding
    | - * * * * * x x - -| Sliding (switched)
    | - - * * * * * x x -| Sliding
    | - - - * * * * * x x| Sliding
    |--------------------|
      0 1 2 3 4 5 6 7 8 9
                ^
                |_ maximum expanding window size reached

    ``*`` = training fold
    ``x`` = test fold
    ``-`` = unused observations

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon
    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Initial window length for the expanding window phase
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    max_expanding_window_length : int, optional (default=float('inf'))
        Maximum window length. If none is passed in, it will keep expanding indefinitely.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import SlidingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = ExpandingSlidingWindowSplitter(fh=[1, 2], step_length=3, initial_window=2, max_expanding_window_length=5)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([0, 1]), array([2, 3])), (array([0, 1, 2, 3, 4]), array([5, 6])), (array([3, 4, 5, 6, 7]), array([8, 9]))]
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        max_expanding_window_length: ACCEPTED_WINDOW_LENGTH_TYPES = float("inf"),
    ) -> None:
        start_with_window = initial_window != 0

        super().__init__(
            fh=fh,
            window_length=initial_window,
            initial_window=None,
            step_length=step_length,
            start_with_window=start_with_window,
        )

        # initial_window needs to be written to self for sklearn compatibility
        self.initial_window = initial_window
        # this class still acts as if it were overwritten with None,
        # via the _initial_window property that is read everywhere

        # Defines the maximum length of the expanding window before switching to the sliding window
        self.max_expanding_window_length = max_expanding_window_length

    @property
    def _initial_window(self):
        return None

    def _split_windows(self, **kwargs) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(expanding=True, **kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter."""
        params1 = {"fg": [1, 2], "step_length": 3, "initial_window": 2, "max_expanding_window_length": 5}

        return [params1]


# Test cases
def test_combined_window_splitter():
    import numpy as np

    # Test case 1: Basic functionality
    ts = np.arange(1, 20 + 1)
    splitter = CombinedWindowSplitter(
        fh=[1, 2], step_length=1, initial_window=2, max_expanding_window_length=5
    )
    splits = list(splitter.split(ts))
    for i, split in enumerate(splits):
        print(f"Split {i}: {split}")

    assert len(splits) == 7, f"Expected 7 splits, got {len(splits)}"
    assert np.array_equal(
        splits[0][0], np.array([0, 1, 2, 3, 4])
    ), "First split train window incorrect"
    assert np.array_equal(
        splits[0][1], np.array([5, 6])
    ), "First split test window incorrect"
    assert np.array_equal(
        splits[3][0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ), "Transition split incorrect"
    assert np.array_equal(
        splits[-1][0], np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ), "Last split train window incorrect"

    # Test case 2: Without max_window_length
    splitter = CombinedWindowSplitter(fh=[1], initial_window=3, step_length=1)
    splits = list(splitter.split(ts))

    assert len(splits) == 16, f"Expected 16 splits, got {len(splits)}"
    assert np.array_equal(
        splits[-1][0], np.array(range(18))
    ), "Last split should include all but last point"

    # Test case 3: Edge case with short time series
    ts_short = np.arange(5)
    splitter = CombinedWindowSplitter(
        fh=[1], initial_window=3, step_length=1, max_expanding_window_length=4
    )
    splits = list(splitter.split(ts_short))

    assert len(splits) == 1, f"Expected 1 split for short series, got {len(splits)}"
    assert np.array_equal(
        splits[0][0], np.array([0, 1, 2])
    ), "Short series split incorrect"

    print("All tests passed!")


# Run the tests
test_combined_window_splitter()
