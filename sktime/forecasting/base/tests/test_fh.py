# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ForecastingHorizon object."""

__author__ = ["mloning", "khrapovs"]

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from numpy.testing._private.utils import assert_array_equal
from pytest import raises

from sktime.datasets import load_airline
from sktime.datatypes._utilities import get_cutoff
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import (
    DELEGATED_METHODS,
    _check_freq,
    _extract_freq_from_cutoff,
)
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import (
    INDEX_TYPE_LOOKUP,
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    VALID_INDEX_FH_COMBINATIONS,
)
from sktime.utils._testing.forecasting import _make_fh, make_forecasting_problem
from sktime.utils._testing.series import _make_index
from sktime.utils.datetime import (
    _coerce_duration_to_int,
    _get_duration,
    _get_freq,
    _get_intervals_count_and_unit,
    _shift,
    infer_freq,
)
from sktime.utils.validation._dependencies import _check_estimator_deps
from sktime.utils.validation.series import is_in_valid_index_types, is_integer_index


def _assert_index_equal(a, b):
    """Compare forecasting horizons."""
    assert isinstance(a, pd.Index)
    assert isinstance(b, pd.Index)
    assert a.equals(b)


@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("steps", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
def test_fh(index_type, fh_type, is_relative, steps):
    """Testing ForecastingHorizon conversions."""
    int_types = ["int64", "int32"]
    steps_is_int = (
        isinstance(steps, (int, np.integer)) or np.array(steps).dtype in int_types
    )
    steps_is_timedelta = isinstance(steps, pd.Timedelta) or (
        isinstance(steps, list) and isinstance(pd.Index(steps), pd.TimedeltaIndex)
    )
    steps_and_fh_incompatible = (fh_type == "timedelta" and steps_is_int) or (
        fh_type != "timedelta" and steps_is_timedelta
    )
    if steps_and_fh_incompatible:
        pytest.skip("steps and fh_type are incompatible")
    # generate data
    y = make_forecasting_problem(index_type=index_type)
    if index_type == "int":
        assert is_integer_index(y.index)
    else:
        assert isinstance(y.index, INDEX_TYPE_LOOKUP.get(index_type))

    # split data
    y_train, y_test = temporal_train_test_split(y, test_size=10)

    # choose cutoff point
    cutoff_idx = get_cutoff(y_train, return_index=True)
    cutoff = cutoff_idx[0]

    # generate fh
    fh = _make_fh(cutoff_idx, steps, fh_type, is_relative)
    # update frequency of the forecasting horizon
    fh.freq = infer_freq(y)
    if fh_type == "int":
        assert is_integer_index(fh.to_pandas())
    else:
        assert isinstance(fh.to_pandas(), INDEX_TYPE_LOOKUP.get(fh_type))

    # get expected outputs
    if isinstance(steps, int):
        steps = np.array([steps])
    elif isinstance(steps, pd.Timedelta):
        steps = pd.Index([steps])
    else:
        steps = pd.Index(steps)

    if steps.dtype in int_types:
        fh_relative = pd.Index(steps, dtype="int64").sort_values()
        fh_absolute = y.index[np.where(y.index == cutoff)[0] + steps].sort_values()
        fh_indexer = fh_relative - 1
    else:
        fh_relative = steps.sort_values()
        fh_absolute = (cutoff + steps).sort_values()
        fh_indexer = None

    if steps.dtype in int_types:
        null = 0
    else:
        null = pd.Timedelta(0)
    fh_oos = fh.to_pandas()[fh_relative > null]
    is_oos = len(fh_oos) == len(fh)
    fh_ins = fh.to_pandas()[fh_relative <= null]
    is_ins = len(fh_ins) == len(fh)

    # check outputs
    # check relative representation
    _assert_index_equal(fh_absolute, fh.to_absolute(cutoff).to_pandas())
    assert not fh.to_absolute(cutoff).is_relative

    # check relative representation
    _assert_index_equal(fh_relative, fh.to_relative(cutoff).to_pandas())
    assert fh.to_relative(cutoff).is_relative

    if steps.dtype in int_types:
        # check index-like representation
        _assert_index_equal(fh_indexer, fh.to_indexer(cutoff))
    else:
        with pytest.raises(NotImplementedError):
            fh.to_indexer(cutoff)

    # check in-sample representation
    # we only compare the numpy array here because the expected solution is
    # formatted in a slightly different way than the generated solution
    np.testing.assert_array_equal(
        fh_ins.to_numpy(), fh.to_in_sample(cutoff).to_pandas()
    )
    assert fh.to_in_sample(cutoff).is_relative == is_relative
    assert fh.is_all_in_sample(cutoff) == is_ins

    # check out-of-sample representation
    np.testing.assert_array_equal(
        fh_oos.to_numpy(), fh.to_out_of_sample(cutoff).to_pandas()
    )
    assert fh.to_out_of_sample(cutoff).is_relative == is_relative
    assert fh.is_all_out_of_sample(cutoff) == is_oos


def test_fh_method_delegation():
    """Test ForecastingHorizon delegated methods."""
    fh = ForecastingHorizon(1)
    for method in DELEGATED_METHODS:
        assert hasattr(fh, method)


BAD_INPUT_ARGS = (
    (1, 2),  # tuple
    "some_string",  # string
    0.1,  # float
    -0.1,  # negative float
    np.array([0.1, 2]),  # float in array
    None,
)


@pytest.mark.parametrize("arg", BAD_INPUT_ARGS)
def test_check_fh_values_bad_input_types(arg):
    """Negative test for bad ForecastingHorizon arguments."""
    with raises(TypeError):
        ForecastingHorizon(arg)


DUPLICATE_INPUT_ARGS = (np.array([1, 2, 2]), [3, 3, 1])


@pytest.mark.parametrize("arg", DUPLICATE_INPUT_ARGS)
def test_check_fh_values_duplicate_input_values(arg):
    """Negative test for ForecastingHorizon input arguments."""
    with raises(ValueError):
        ForecastingHorizon(arg)


GOOD_ABSOLUTE_INPUT_ARGS = (
    pd.Index([1, 2, 3]),
    pd.period_range("2000-01-01", periods=3, freq="D"),
    pd.date_range("2000-01-01", periods=3, freq="M"),
    np.array([1, 2, 3]),
    [1, 2, 3],
    1,
)


@pytest.mark.parametrize("arg", GOOD_ABSOLUTE_INPUT_ARGS)
def test_check_fh_absolute_values_input_conversion_to_pandas_index(arg):
    """Test conversion of absolute horizons to pandas index."""
    assert is_in_valid_index_types(
        ForecastingHorizon(arg, is_relative=False).to_pandas()
    )


GOOD_RELATIVE_INPUT_ARGS = [
    pd.timedelta_range(pd.to_timedelta(1, unit="D"), periods=3, freq="D"),
    [np.timedelta64(x, "D") for x in range(3)],
    [timedelta(days=x) for x in range(3)],
]


@pytest.mark.parametrize("arg", GOOD_RELATIVE_INPUT_ARGS)
def test_check_fh_relative_values_input_conversion_to_pandas_index(arg):
    """Test conversion of relative horizons to pandas index."""
    output = ForecastingHorizon(arg, is_relative=True).to_pandas()
    assert is_in_valid_index_types(output)


TIMEPOINTS = [
    pd.Period("2000", freq="M"),
    pd.Timestamp("2000-01-01").to_period(freq="D"),
    int(1),
    3,
]

LENGTH1_INDICES = [pd.Index([x]) for x in TIMEPOINTS]
LENGTH1_INDICES += [pd.DatetimeIndex(["2000-01-01"], freq="D")]


@pytest.mark.parametrize("timepoint", TIMEPOINTS)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift(timepoint, by):
    """Test shifting of cutoff index element."""
    ret = _shift(timepoint, by=by)

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert timepoint == ret


@pytest.mark.parametrize("timepoint", LENGTH1_INDICES)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift_index(timepoint, by):
    """Test shifting of cutoff index, length 1 pandas.Index type."""
    ret = _shift(timepoint, by=by, return_index=True)

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert (timepoint == ret).all()


DURATIONS_ALLOWED = [
    pd.TimedeltaIndex(range(3), unit="D", freq="D"),
    pd.TimedeltaIndex(range(0, 9, 3), unit="D", freq="3D"),
    pd.tseries.offsets.MonthEnd(3),
    # we also support pd.Timedelta, but it does not have freqstr so we
    # cannot automatically infer the unit during testing
    # pd.Timedelta(days=3, freq="D"),
]
DURATIONS_NOT_ALLOWED = [
    pd.Index(pd.tseries.offsets.Day(day) for day in range(3)),
    # we also support pd.Timedelta, but it does not have freqstr so we
    # cannot automatically infer the unit during testing
    # pd.Timedelta(days=3, freq="D"),
]


@pytest.mark.parametrize("duration", DURATIONS_ALLOWED)
def test_coerce_duration_to_int(duration):
    """Test coercion of duration to int."""
    ret = _coerce_duration_to_int(duration, freq=_get_freq(duration))

    # check output type is always integer
    assert (type(ret) in (np.integer, int)) or is_integer_index(ret)

    # check result
    if isinstance(duration, pd.Index):
        np.testing.assert_array_equal(ret, range(3))

    if isinstance(duration, pd.tseries.offsets.BaseOffset):
        assert ret == 1


@pytest.mark.parametrize("duration", DURATIONS_NOT_ALLOWED)
def test_coerce_duration_to_int_with_non_allowed_durations(duration):
    """Test coercion of duration to int."""
    with pytest.raises(ValueError, match="frequency is missing"):
        _coerce_duration_to_int(duration, freq=_get_freq(duration))


@pytest.mark.parametrize("n_timepoints", [3, 5])
@pytest.mark.parametrize("index_type", INDEX_TYPE_LOOKUP.keys())
def test_get_duration(n_timepoints, index_type):
    """Test getting of duration."""
    if index_type != "timedelta":
        index = _make_index(n_timepoints, index_type)
        duration = _get_duration(index)
        # check output type is duration type
        assert isinstance(
            duration, (pd.Timedelta, pd.tseries.offsets.BaseOffset, int, np.integer)
        )

        # check integer output
        duration = _get_duration(index, coerce_to_int=True)
        assert isinstance(duration, (int, np.integer))
        assert duration == n_timepoints - 1
    else:
        match = "index_class: timedelta is not supported"
        with pytest.raises(ValueError, match=match):
            _make_index(n_timepoints, index_type)


FIXED_FREQUENCY_STRINGS = ["10T", "H", "D", "2D"]
NON_FIXED_FREQUENCY_STRINGS = ["W-WED", "W-SUN", "W-SAT", "M"]
FREQUENCY_STRINGS = [*FIXED_FREQUENCY_STRINGS, *NON_FIXED_FREQUENCY_STRINGS]


@pytest.mark.parametrize("freqstr", FREQUENCY_STRINGS)
def test_to_absolute_freq(freqstr):
    """Test conversion when anchorings included in frequency."""
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    fh = ForecastingHorizon([1, 2, 3])

    abs_fh = fh.to_absolute(cutoff)
    assert abs_fh._values.freqstr == freqstr


@pytest.mark.parametrize("freqstr", FREQUENCY_STRINGS)
def test_absolute_to_absolute_with_integer_horizon(freqstr):
    """Test converting between absolute and relative with integer horizon."""
    # Converts from absolute to relative and back to absolute
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    fh = ForecastingHorizon([1, 2, 3])
    abs_fh = fh.to_absolute(cutoff)

    converted_abs_fh = abs_fh.to_relative(cutoff).to_absolute(cutoff)
    assert_array_equal(abs_fh, converted_abs_fh)
    assert converted_abs_fh._values.freqstr == freqstr


@pytest.mark.parametrize("freqstr", FIXED_FREQUENCY_STRINGS)
def test_absolute_to_absolute_with_timedelta_horizon(freqstr):
    """Test converting between absolute and relative."""
    # Converts from absolute to relative and back to absolute
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    count, unit = _get_intervals_count_and_unit(freq=freqstr)
    fh = ForecastingHorizon(
        pd.timedelta_range(pd.to_timedelta(count, unit=unit), freq=freqstr, periods=3)
    )
    abs_fh = fh.to_absolute(cutoff)

    converted_abs_fh = abs_fh.to_relative(cutoff).to_absolute(cutoff)
    assert_array_equal(abs_fh, converted_abs_fh)
    assert converted_abs_fh._values.freqstr == freqstr


@pytest.mark.parametrize("freqstr", FREQUENCY_STRINGS)
def test_relative_to_relative_with_integer_horizon(freqstr):
    """Test converting between relative and absolute with integer horizons."""
    # Converts from relative to absolute and back to relative
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    fh = ForecastingHorizon([1, 2, 3])
    abs_fh = fh.to_absolute(cutoff)

    converted_rel_fh = abs_fh.to_relative(cutoff)
    assert_array_equal(fh, converted_rel_fh)


@pytest.mark.parametrize("freqstr", FIXED_FREQUENCY_STRINGS)
def test_relative_to_relative_with_timedelta_horizon(freqstr):
    """Test converting between relative and absolute with timedelta horizons."""
    # Converts from relative to absolute and back to relative
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    count, unit = _get_intervals_count_and_unit(freq=freqstr)
    fh = ForecastingHorizon(
        pd.timedelta_range(pd.to_timedelta(count, unit=unit), freq=freqstr, periods=3)
    )
    abs_fh = fh.to_absolute(cutoff)

    converted_rel_fh = abs_fh.to_relative(cutoff)
    assert_array_equal(converted_rel_fh, np.arange(1, 4))


@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_to_relative(freq: str):
    """Test conversion to relative.

    Fixes bug in
    https://github.com/sktime/sktime/issues/1935#issue-1114814142
    """
    freq = "2H"
    t = pd.date_range(start="2021-01-01", freq=freq, periods=5)
    cutoff = get_cutoff(t, return_index=True, reverse_order=True)
    fh_abs = ForecastingHorizon(t, is_relative=False)
    fh_rel = fh_abs.to_relative(cutoff=cutoff)
    assert_array_equal(fh_rel, np.arange(5))


@pytest.mark.parametrize("idx", range(5))
@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_to_absolute_int(idx: int, freq: str):
    """Test converting between relative and absolute."""
    # Converts from relative to absolute and back to relative
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freq, periods=5))
    fh = ForecastingHorizon([1, 2, 3])
    cutoff = train.index[[idx]]
    cutoff.freq = train.index.freq
    absolute_int = fh.to_absolute_int(start=train.index[0], cutoff=cutoff)
    assert_array_equal(fh + idx, absolute_int)


@pytest.mark.parametrize("idx", range(5))
@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_to_absolute_int_fh_with_freq(idx: int, freq: str):
    """Test converting between relative and absolute, freq passed to fh."""
    # Converts from relative to absolute and back to relative
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freq, periods=5))
    fh = ForecastingHorizon([1, 2, 3], freq=freq)
    cutoff = train.index[idx]
    absolute_int = fh.to_absolute_int(start=train.index[0], cutoff=cutoff)
    assert_array_equal(fh + idx, absolute_int)


@pytest.mark.skipif(
    not _check_estimator_deps(AutoETS, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("freqstr", ["W-WED", "W-SUN", "W-SAT"])
def test_estimator_fh(freqstr):
    """Test model fitting with anchored frequency."""
    train = pd.Series(
        np.random.uniform(low=2000, high=7000, size=(104,)),
        index=pd.date_range("2019-01-02", freq=freqstr, periods=104),
    )
    forecaster = AutoETS(auto=True, sp=52, n_jobs=-1, restrict=True)
    forecaster.fit(train)
    fh = ForecastingHorizon(np.arange(1, 27))
    pred = forecaster.predict(fh)
    expected_fh = fh.to_absolute(train.index[-1])
    assert_array_equal(pred.index.to_numpy(), expected_fh.to_numpy())


@pytest.mark.parametrize("freq", ["G", "W1"])
def test_error_with_incorrect_string_frequency(freq: str):
    """Test error with incorrect string frequency string."""
    match = f"Invalid frequency: {freq}"
    with pytest.raises(ValueError, match=match):
        ForecastingHorizon([1, 2, 3], freq=freq)
    fh = ForecastingHorizon([1, 2, 3])
    with pytest.raises(ValueError, match=match):
        fh.freq = freq


@pytest.mark.parametrize("freqstr", ["M", "D"])
def test_frequency_setter(freqstr):
    """Test frequency setter."""
    fh = ForecastingHorizon([1, 2, 3])
    assert fh.freq is None

    fh.freq = freqstr
    assert fh.freq == freqstr

    fh = ForecastingHorizon([1, 2, 3], freq=freqstr)
    assert fh.freq == freqstr


# TODO: Replace this long running test with fast unit test
@pytest.mark.skipif(
    not _check_estimator_deps(AutoETS, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_auto_ets():
    """Test failure case from #1435.

    https://github.com/sktime/sktime/issues/1435#issue-1000175469
    """
    freq = "30T"
    _y = np.arange(50) + np.random.rand(50) + np.sin(np.arange(50) / 4) * 10
    t = pd.date_range("2021-09-19", periods=50, freq=freq)
    y = pd.Series(_y, index=t)
    y.index = y.index.to_period(freq=freq)
    forecaster = AutoETS(sp=12, auto=True, n_jobs=-1)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3])
    pd.testing.assert_index_equal(
        y_pred.index,
        pd.date_range("2021-09-19", periods=53, freq=freq)[-3:].to_period(freq=freq),
    )


# TODO: Replace this long running test with fast unit test
@pytest.mark.skipif(
    not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_exponential_smoothing():
    """Test failure case from #1876.

    https://github.com/sktime/sktime/issues/1876#issue-1103752402.
    """
    y = load_airline()
    # Change index to 10 min interval
    freq = "10Min"
    time_range = pd.date_range(
        pd.to_datetime("2019-01-01 00:00"),
        pd.to_datetime("2019-01-01 23:55"),
        freq=freq,
    )
    # Period Index does not work
    y.index = time_range.to_period()

    forecaster = ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)
    forecaster.fit(y, fh=[1, 2, 3, 4, 5, 6])
    y_pred = forecaster.predict()
    pd.testing.assert_index_equal(
        y_pred.index, pd.period_range("2019-01-02 00:00", periods=6, freq=freq)
    )


# TODO: Replace this long running test with fast unit test
@pytest.mark.skipif(
    not _check_estimator_deps(AutoARIMA, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_auto_arima():
    """Test failure case from #805.

    https://github.com/sktime/sktime/issues/805#issuecomment-891848228.
    """
    time_index = pd.date_range("January 1, 2021", periods=8, freq="1D")
    X = pd.DataFrame(
        np.random.randint(0, 4, 24).reshape(8, 3),
        columns=["First", "Second", "Third"],
        index=time_index,
    )
    y = pd.Series([1, 3, 2, 4, 5, 2, 3, 1], index=time_index)

    fh_ = ForecastingHorizon(X.index[5:], is_relative=False)

    a_clf = AutoARIMA(start_p=2, start_q=2, max_p=5, max_q=5)
    clf = a_clf.fit(X=X[:5], y=y[:5])
    y_pred_sk = clf.predict(fh=fh_, X=X[5:])

    pd.testing.assert_index_equal(
        y_pred_sk.index, pd.date_range("January 6, 2021", periods=3, freq="1D")
    )

    time_index = pd.date_range("January 1, 2021", periods=8, freq="2D")
    X = pd.DataFrame(
        np.random.randint(0, 4, 24).reshape(8, 3),
        columns=["First", "Second", "Third"],
        index=time_index,
    )
    y = pd.Series([1, 3, 2, 4, 5, 2, 3, 1], index=time_index)

    fh = ForecastingHorizon(X.index[5:], is_relative=False)

    a_clf = AutoARIMA(start_p=2, start_q=2, max_p=5, max_q=5)
    clf = a_clf.fit(X=X[:5], y=y[:5])
    y_pred_sk = clf.predict(fh=fh, X=X[5:])

    pd.testing.assert_index_equal(
        y_pred_sk.index, pd.date_range("January 11, 2021", periods=3, freq="2D")
    )


def test_extract_freq_from_inputs() -> None:
    """Test extract frequency from inputs."""
    assert _check_freq(None) is None
    cutoff = pd.Period("2020", freq="D")
    assert _check_freq(cutoff) == "D"
    assert _check_freq("D") == "D"


@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_extract_freq_from_cutoff(freq: str) -> None:
    """Test extract frequency from cutoff."""
    assert _extract_freq_from_cutoff(pd.Period("2020", freq=freq)) == freq


@pytest.mark.parametrize("x", [1, pd.Timestamp("2020")])
def test_extract_freq_from_cutoff_with_wrong_input(x) -> None:
    """Test extract frequency from cutoff with wrong input."""
    assert _extract_freq_from_cutoff(x) is None
