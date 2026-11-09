"""Regression tests for variable-arity splitting and window composition."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import (
    ExpandingWindowResampler,
    ExpandingWindowSplitter,
    ResamplerToSplitter,
    SameLocSplitter,
    SlidingWindowResampler,
    SlidingWindowSplitter,
)
from sktime.split import TestPlusTrainSplitter as _TestPlusTrainSplitter
from sktime.split.base import BaseSplitter
from sktime.split.compose import Repeat, SyncToLongest
from sktime.utils.validation.forecasting import check_cv


@pytest.mark.parametrize("cls", [SlidingWindowResampler, ExpandingWindowResampler])
@pytest.mark.parametrize("kind", ["series", "frame", "array", "datetime", "period"])
def test_resampler_index_and_data_agree(cls, kind):
    """Public methods agree and the last full window includes the final row."""
    y = pd.Series(np.arange(7.0), name="value")
    if kind == "frame":
        y = y.to_frame()
    elif kind == "array":
        y = y.to_numpy().reshape(-1, 1)
    elif kind == "datetime":
        y.index = pd.date_range("2020-01-01", periods=7)
    elif kind == "period":
        y.index = pd.period_range("2020-01", periods=7, freq="M")
    cv = cls(3, 2)
    positions = list(cv.split(y))
    labels = list(cv.split_loc(y))
    data = list(cv.split_series(y))
    assert len(positions) == len(labels) == len(data) == cv.get_n_splits(y) == 3
    assert positions[-1][-1] == 6
    assert cv.get_fh() is None
    np.testing.assert_array_equal(cv.get_cutoffs(y), [2, 4, 6])
    for pos, loc, values in zip(positions, labels, data):
        expected_pos = (
            np.arange(pos[-1] + 1) if cls is ExpandingWindowResampler else pos
        )
        np.testing.assert_array_equal(pos, expected_pos)
        if isinstance(y, np.ndarray):
            np.testing.assert_array_equal(values, y[pos])
        else:
            pd.testing.assert_index_equal(loc, y.index[pos])
            assert values.equals(y.iloc[pos])


@pytest.mark.parametrize("cls", [SlidingWindowResampler, ExpandingWindowResampler])
def test_windows_and_empty_results(cls):
    """Window contents are exact and too-short series yield no full windows."""
    actual = [x.tolist() for x in cls(3, 2).split(pd.RangeIndex(7))]
    expected = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    if cls is ExpandingWindowResampler:
        expected = [list(range(end)) for end in [3, 5, 7]]
    assert actual == expected
    assert list(cls(8).split(pd.RangeIndex(7))) == []


@pytest.mark.parametrize("name", ["window_length", "step_length"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_invalid_window_parameters(name, value):
    """Reject invalid sizes rather than silently coercing them."""
    with pytest.raises(ValueError, match=name):
        SlidingWindowResampler(**{name: value})


@pytest.mark.parametrize("cls", [SlidingWindowResampler, ExpandingWindowResampler])
def test_hierarchical_windows(cls):
    """Broadcast window positions to every series, retaining all index levels."""
    index = pd.MultiIndex.from_product(
        [["group"], ["a", "b"], pd.date_range("2020-01-01", periods=7)],
        names=["group", "instance", "time"],
    )
    y = pd.DataFrame({"value": np.arange(14.0)}, index=index)
    original = y.copy(deep=True)
    cv = cls(3, 2)
    for i, (pos, loc, values) in enumerate(
        zip(cv.split(y), cv.split_loc(y), cv.split_series(y))
    ):
        end = 3 + 2 * i
        start = 0 if cls is ExpandingWindowResampler else end - 3
        expected = np.r_[np.arange(start, end), np.arange(start + 7, end + 7)]
        np.testing.assert_array_equal(pos, expected)
        pd.testing.assert_index_equal(loc, y.index[expected])
        pd.testing.assert_frame_equal(values, y.iloc[expected])
    pd.testing.assert_frame_equal(y, original)


@pytest.mark.parametrize(
    "resampler,splitter",
    [
        (SlidingWindowResampler, SlidingWindowSplitter),
        (ExpandingWindowResampler, ExpandingWindowSplitter),
    ],
)
def test_composed_windows_match_existing_splitters(resampler, splitter):
    """Adding a horizon reproduces established train/test window semantics."""
    y = pd.Series(np.arange(15.0))
    cv = ResamplerToSplitter(resampler(5, 2), fh=[1, 3])
    kwargs = (
        {"window_length": 5}
        if splitter is SlidingWindowSplitter
        else {"initial_window": 5}
    )
    reference = splitter(fh=[1, 3], step_length=2, **kwargs)
    for actual, expected in zip(cv.split(y), reference.split(y)):
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    assert cv.get_n_splits(y) == reference.get_n_splits(y)
    np.testing.assert_array_equal(cv.get_cutoffs(y), reference.get_cutoffs(y))
    result = evaluate(NaiveForecaster(), cv=cv, y=y, error_score="raise")
    assert len(result) == cv.get_n_splits(y)


def test_cv_rejects_uncomposed_resampler():
    """Forecast evaluation rejects one-fold outputs with an actionable error."""
    with pytest.raises(ValueError, match="ResamplerToSplitter"):
        check_cv(SlidingWindowResampler())


class _ThreeFoldSplitter(BaseSplitter):
    """Minimal extension exercising higher arities and negative index filtering."""

    _tags = {"split:arity": 3}

    def _split(self, y):
        yield np.array([-1, 0]), np.array([1]), np.array([2])


def test_three_fold_extension():
    """Location and data slicing preserve arbitrary fold counts."""
    y = pd.Series([10.0, 20.0, 30.0], index=pd.RangeIndex(10, 13))
    cv = _ThreeFoldSplitter()
    assert [x.tolist() for x in next(cv.split(y))] == [[0], [1], [2]]
    assert [x.tolist() for x in next(cv.split_loc(y))] == [[10], [11], [12]]
    assert [x.tolist() for x in next(cv.split_series(y))] == [[10], [20], [30]]


@pytest.mark.parametrize("mode", ["entry", "sequence"])
@pytest.mark.parametrize("random_repeat", [False, True])
def test_repeated_resampling(mode, random_repeat):
    """Repetition preserves a single-fold output rather than unpacking it."""
    y = pd.Series(np.arange(7.0))
    cv = Repeat(
        SlidingWindowResampler(3, 2), times=2, mode=mode, random_repeat=random_repeat
    )
    actual = [x.tolist() for x in cv.split(y)]
    windows = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    expected = [x for w in windows for x in [w, w]] if mode == "entry" else windows * 2
    assert actual == expected
    assert cv.get_tag("split:arity") == 1
    assert cv.get_fh() is None
    assert cv.get_n_splits(y) == len(actual)


def test_sameloc_resampling():
    """Location composition retains one-fold semantics on a larger index."""
    y = pd.Series(np.arange(10.0), index=pd.RangeIndex(10, 20))
    cv = SameLocSplitter(SlidingWindowResampler(3, 2), y_template=y.iloc[2:9])
    assert cv.get_tag("split:arity") == 1
    assert [x.tolist() for x in cv.split(y)] == [[2, 3, 4], [4, 5, 6], [6, 7, 8]]


@pytest.mark.parametrize("cls", [_TestPlusTrainSplitter, SyncToLongest])
def test_pair_only_compositions_reject_single_windows(cls):
    """Compositions that specifically need train and test folds explain misuse."""
    with pytest.raises(ValueError, match="requires a train/test splitter"):
        cls(SlidingWindowResampler())
