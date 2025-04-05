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
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import (
    INDEX_TYPE_LOOKUP,
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    VALID_INDEX_FH_COMBINATIONS,
)
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.forecasting import _make_fh, make_forecasting_problem
from sktime.utils._testing.series import _make_index, _make_series
from sktime.utils.datetime import (
    _coerce_duration_to_int,
    _get_duration,
    _get_freq,
    _get_intervals_count_and_unit,
    _shift,
    infer_freq,
)
from sktime.utils.dependencies import _check_estimator_deps, _check_soft_dependencies
from sktime.utils.validation.series import is_in_valid_index_types, is_integer_index


def _assert_index_equal(a, b):
    """Compare forecasting horizons."""
    assert isinstance(a, pd.Index)
    assert isinstance(b, pd.Index)
    assert a.equals(b)


# fixtures
@pytest.fixture(params=None)
def good_absolute_input_arg(request):
    """Parametrized by integer values."""
    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        m_freq = "ME"
    else:
        m_freq = "M"

    params = [
        pd.Index([1, 2, 3]),
        pd.period_range("2000-01-01", periods=3, freq="D"),
        pd.date_range("2000-01-01", periods=3, freq=m_freq),
        np.array([1, 2, 3]),
        [1, 2, 3],
        1,
    ]
    return params[request.param]


@pytest.fixture(params=None)
def good_relative_input_arg(request):
    """Parametrized by integer values."""
    params = [
        pd.timedelta_range(pd.to_timedelta(1, unit="D"), periods=3, freq="D"),
        [np.timedelta64(x, "D") for x in range(3)],
        [timedelta(days=x) for x in range(3)],
    ]
    return params[request.param]


@pytest.fixture(params=None)
def timepoint(request):
    """Parametrized by integer values."""
    TIMEPOINTS = [
        pd.Period("2000", freq="M"),
        pd.Timestamp("2000-01-01").to_period(freq="D"),
        1,
        3,
    ]
    return TIMEPOINTS[request.param]


@pytest.fixture(params=None)
def length1_index(request):
    """Parametrized by integer values."""
    TIMEPOINTS = [
        pd.Period("2000", freq="M"),
        pd.Timestamp("2000-01-01").to_period(freq="D"),
        1,
        3,
    ]
    LENGTH1_INDICES = [pd.Index([x]) for x in TIMEPOINTS]
    LENGTH1_INDICES += [pd.DatetimeIndex(["2000-01-01"], freq="D")]
    return LENGTH1_INDICES[request.param]


# test generate hook to ensure fixture creation happens only on test time,
# not on module load time
# this is because pandas fixtures may break, they contain risky freq args
def pytest_generate_tests(metafunc):
    if "good_absolute_input_arg" in metafunc.fixturenames:
        metafunc.parametrize("good_absolute_input_arg", range(6), indirect=True)
    if "good_relative_input_arg" in metafunc.fixturenames:
        metafunc.parametrize("good_relative_input_arg", range(3), indirect=True)
    if "timepoint" in metafunc.fixturenames:
        metafunc.parametrize("timepoint", range(4), indirect=True)
    if "length1_index" in metafunc.fixturenames:
        metafunc.parametrize("length1_index", range(5), indirect=True)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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
    _assert_index_equal(fh_absolute, fh.to_absolute_index(cutoff))
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("arg", BAD_INPUT_ARGS)
def test_check_fh_values_bad_input_types(arg):
    """Negative test for bad ForecastingHorizon arguments."""
    with raises(TypeError):
        ForecastingHorizon(arg)


DUPLICATE_INPUT_ARGS = (np.array([1, 2, 2]), [3, 3, 1])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("arg", DUPLICATE_INPUT_ARGS)
def test_check_fh_values_duplicate_input_values(arg):
    """Negative test for ForecastingHorizon input arguments."""
    with raises(ValueError):
        ForecastingHorizon(arg)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_check_fh_absolute_input_conversion_to_pandas_index(good_absolute_input_arg):
    """Test conversion of absolute horizons to pandas index."""
    arg = good_absolute_input_arg
    assert is_in_valid_index_types(
        ForecastingHorizon(arg, is_relative=False).to_pandas()
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_check_fh_relative_values_conversion_to_pandas_index(good_relative_input_arg):
    """Test conversion of relative horizons to pandas index."""
    arg = good_relative_input_arg
    output = ForecastingHorizon(arg, is_relative=True).to_pandas()
    assert is_in_valid_index_types(output)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift_index(length1_index, by):
    """Test shifting of cutoff index, length 1 pandas.Index type."""
    timepoint = length1_index
    ret = _shift(timepoint, by=by, return_index=True)

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert (timepoint == ret).all()


DURATIONS_ALLOWED = [
    pd.TimedeltaIndex(pd.to_timedelta(range(3), unit="D"), freq="D"),
    pd.TimedeltaIndex(pd.to_timedelta(range(0, 9, 3), unit="D"), freq="3D"),
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("duration", DURATIONS_NOT_ALLOWED)
def test_coerce_duration_to_int_with_non_allowed_durations(duration):
    """Test coercion of duration to int."""
    with pytest.raises(ValueError, match="frequency is missing"):
        _coerce_duration_to_int(duration, freq=_get_freq(duration))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


FIXED_FREQUENCY_STRINGS = ["10min", "H", "D", "2D"]
NON_FIXED_FREQUENCY_STRINGS = ["W-WED", "W-SUN", "W-SAT", "M"]
FREQUENCY_STRINGS = [*FIXED_FREQUENCY_STRINGS, *NON_FIXED_FREQUENCY_STRINGS]


def _get_expected_freqstr(freqstr):
    # special case for 10min, T is being deprecated and replaced by min
    if _check_soft_dependencies("pandas<2.2.0", severity="none"):
        if freqstr == "10min":
            return "10T"
        return freqstr
    # on more recent pandas versions, >=2.2.0
    if freqstr == "H":
        return "h"
    if freqstr == "M":
        return "ME"
    return freqstr


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("freqstr", FREQUENCY_STRINGS)
def test_to_absolute_freq(freqstr):
    """Test conversion when anchorings included in frequency."""
    train = pd.Series(1, index=pd.date_range("2021-10-06", freq=freqstr, periods=3))
    cutoff = get_cutoff(train, return_index=True)
    fh = ForecastingHorizon([1, 2, 3])

    abs_fh = fh.to_absolute(cutoff)
    assert abs_fh._values.freqstr == _get_expected_freqstr(freqstr)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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
    fh_freqstr = converted_abs_fh._values.freqstr
    assert fh_freqstr == _get_expected_freqstr(freqstr)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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

    assert converted_abs_fh._values.freqstr == _get_expected_freqstr(freqstr)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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


@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_to_absolute_with_multiple_freq(freq: str):
    """Test to_absolute with multiple freq"""
    fh = ForecastingHorizon([0, 1, 2, 3, 4], is_relative=True)
    start = "2024-09-26 17:24"
    cutoff = pd.PeriodIndex([start], freq=freq)
    absolute = fh.to_absolute(cutoff)
    date_range = pd.date_range(start=start, freq=freq, periods=5)
    period_index = date_range.to_period(freq)
    assert_array_equal(period_index.to_numpy(), absolute.to_numpy())


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("freqstr", ["W-WED", "W-SUN", "W-SAT"])
def test_estimator_fh(freqstr):
    """Test model fitting with anchored frequency."""
    train = pd.Series(
        np.random.uniform(low=2000, high=7000, size=(104,)),
        index=pd.date_range("2019-01-02", freq=freqstr, periods=104),
    )
    forecaster = NaiveForecaster()
    forecaster.fit(train)
    fh = ForecastingHorizon(np.arange(1, 27))
    pred = forecaster.predict(fh)
    expected_fh = fh.to_absolute(train.index[-1])
    assert_array_equal(pred.index.to_numpy(), expected_fh.to_numpy())


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("freq", ["G", "W1"])
def test_error_with_incorrect_string_frequency(freq: str):
    """Test error with incorrect string frequency string."""
    match = f"Invalid frequency: {freq}"
    with pytest.raises(ValueError, match=match):
        ForecastingHorizon([1, 2, 3], freq=freq)
    fh = ForecastingHorizon([1, 2, 3])
    with pytest.raises(ValueError, match=match):
        fh.freq = freq


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
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
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_estimator_deps(AutoETS, severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_auto_ets_case_with_naive():
    """Test failure case from #1435.

    AutoETS is replaced by NaiveForecaster.

    https://github.com/sktime/sktime/issues/1435#issue-1000175469
    """
    freq = "30T"
    _y = np.arange(50) + np.random.rand(50) + np.sin(np.arange(50) / 4) * 10
    t = pd.date_range("2021-09-19", periods=50, freq=freq)
    y = pd.Series(_y, index=t)
    y.index = y.index.to_period(freq=freq)
    forecaster = NaiveForecaster()
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3])
    pd.testing.assert_index_equal(
        y_pred.index,
        pd.date_range("2021-09-19", periods=53, freq=freq)[-3:].to_period(freq=freq),
    )


# TODO: Replace this long running test with fast unit test
@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_exponential_smoothing_case_with_naive():
    """Test failure case from #1876.

    ExponentialSmoothing is replaced by NaiveForecaster.

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

    forecaster = NaiveForecaster()
    forecaster.fit(y, fh=[1, 2, 3, 4, 5, 6])
    y_pred = forecaster.predict()
    pd.testing.assert_index_equal(
        y_pred.index, pd.period_range("2019-01-02 00:00", periods=6, freq=freq)
    )


# TODO: Replace this long running test with fast unit test
# todo 0.38.0: check whether numpy 2 bound is still necessary
@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_estimator_deps(AutoARIMA, severity="none")
    or _check_soft_dependencies("numpy>=2.0", severity="none"),
    reason="run only if base module has changed or datatypes module has changed",
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_auto_arima_case_with_naive():
    """Test failure case from #805.

    AutoARIMA is replaced by NaiveForecaster.

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

    a_clf = NaiveForecaster()
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

    a_clf = NaiveForecaster()
    clf = a_clf.fit(X=X[:5], y=y[:5])
    y_pred_sk = clf.predict(fh=fh, X=X[5:])

    pd.testing.assert_index_equal(
        y_pred_sk.index, pd.date_range("January 11, 2021", periods=3, freq="2D")
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_extract_freq_from_inputs() -> None:
    """Test extract frequency from inputs."""
    assert _check_freq(None) is None
    cutoff = pd.Period("2020", freq="D")
    assert _check_freq(cutoff) == "D"
    assert _check_freq("D") == "D"


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("freq", FREQUENCY_STRINGS)
def test_extract_freq_from_cutoff(freq: str) -> None:
    """Test extract frequency from cutoff."""
    assert _extract_freq_from_cutoff(pd.Period("2020", freq=freq)) == freq


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("x", [1, pd.Timestamp("2020")])
def test_extract_freq_from_cutoff_with_wrong_input(x) -> None:
    """Test extract frequency from cutoff with wrong input."""
    assert _extract_freq_from_cutoff(x) is None


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_regular_spaced_fh_of_different_periodicity():
    """Test for failure condition from bug #4462.

    Due to pandas frequency inference logic, a specific case of constructing
    `ForecastingHorizon` could upset the constructor: passing a regular `DatetimeIndex`
    with frequency different from the `freq` argument, which would be triggered in some
    `to_absolute` conversions.
    """
    y = _make_series(n_columns=1)

    naive = NaiveForecaster()
    naive.fit(y)
    naive.predict([1, 3, 5])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_standard_range_in_fh():
    """Test using most common ``range`` without start/step."""
    standard_range = ForecastingHorizon(values=range(1, 5 + 1))
    assert (standard_range == ForecastingHorizon(values=[1, 2, 3, 4, 5])).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_range_with_positive_step_in_fh():
    """Test using ``range`` with positive step."""
    range_with_positive_step = ForecastingHorizon(values=range(0, 5, 2))
    assert (range_with_positive_step == ForecastingHorizon(values=[0, 2, 4])).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_range_with_negative_step_in_fh():
    """Test using ``range`` with negative step."""
    range_with_negative_step = ForecastingHorizon(values=range(3, -5, -2))
    assert (range_with_negative_step == ForecastingHorizon(values=[3, 1, -1, -3])).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_range_sorting_in_fh():
    """Test that ``range`` is independent of order."""
    standard_range = ForecastingHorizon(values=range(5))
    assert (standard_range == ForecastingHorizon(values=[0, 3, 4, 1, 2])).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_empty_range_in_fh():
    """Test when ``range`` has zero length."""
    empty_range = ForecastingHorizon(values=range(-5))
    assert (empty_range == ForecastingHorizon(values=[])).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_fh_expected_pred():
    """Test for expected prediction index method."""
    fh = ForecastingHorizon([1, 2, 3])
    y_pred_idx = fh.get_expected_pred_idx(pd.Index([2, 3, 4]))

    assert y_pred_idx.equals(pd.Index([5, 6, 7]))

    y_df = pd.DataFrame([1, 2, 3], index=[2, 3, 4])
    y_pred_idx = fh.get_expected_pred_idx(y_df)

    assert y_pred_idx.equals(pd.Index([5, 6, 7]))

    # pd.MultiIndex case, 2 levels
    idx = pd.MultiIndex.from_tuples([("a", 3), ("a", 5), ("b", 4), ("b", 5), ("b", 6)])
    y_pred_idx = fh.get_expected_pred_idx(idx)

    y_pred_idx_expected = pd.MultiIndex.from_tuples(
        [("a", 6), ("a", 7), ("a", 8), ("b", 7), ("b", 8), ("b", 9)]
    )
    assert y_pred_idx.equals(y_pred_idx_expected)

    y_pred_idx = fh.get_expected_pred_idx(idx, sort_by_time=True)
    y_pred_idx_expected = pd.MultiIndex.from_tuples(
        [("a", 6), ("a", 7), ("b", 7), ("a", 8), ("b", 8), ("b", 9)]
    )
    assert y_pred_idx.equals(y_pred_idx_expected)

    # pd.MultiIndex case, 3 levels
    idx = pd.MultiIndex.from_tuples(
        [("a", 3, 4), ("a", 3, 5), ("b", 5, 4), ("b", 5, 5), ("b", 5, 6)]
    )
    y_pred_idx = fh.get_expected_pred_idx(idx)

    y_pred_idx_expected = pd.MultiIndex.from_tuples(
        [("a", 3, 6), ("a", 3, 7), ("a", 3, 8), ("b", 5, 7), ("b", 5, 8), ("b", 5, 9)]
    )
    assert y_pred_idx.equals(y_pred_idx_expected)

    y_pred_idx = fh.get_expected_pred_idx(idx, sort_by_time=True)

    y_pred_idx_expected = pd.MultiIndex.from_tuples(
        [("a", 3, 6), ("a", 3, 7), ("b", 5, 7), ("a", 3, 8), ("b", 5, 8), ("b", 5, 9)]
    )
    assert y_pred_idx.equals(y_pred_idx_expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_tz_preserved():
    """Test that time zone information is preserved in to_absolute.

    Failure case in issue #5584.
    """
    cutoff = pd.Timestamp("2020-01-01", tz="utc")
    fh_absolute = ForecastingHorizon(range(100), freq="h").to_absolute(cutoff)

    assert fh_absolute[0].tz == cutoff.tz


# the "XE" frequencies are not supported by pandas 1 or 2.0.X
FREQ_STR_FOR_PD22 = ["Y", "2Y", "M", "3M"]

if _check_soft_dependencies("pandas>=2.1.0", severity="none"):
    FREQ_STR_FOR_PD22 += [
        "YE",
        "2YE",
        "ME",
        "3ME",
        "MS",
        "3MS",
        "QS",
        "3QS",
        "YS",
        "3YS",
    ]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("freq", FREQ_STR_FOR_PD22)
def test_pandas22_freq(freq):
    """Test that to_absolute and to_relative conversions work with all freqs.

    Failure case in bug #6499.
    """
    fh = ForecastingHorizon([1, 2, 3])

    datetime_ = pd.date_range("1/1/1870", periods=20, freq=freq)
    cutoff = datetime_[[-1]]
    cutoff.freq = datetime_.freq

    fh.to_absolute(cutoff)  # failure 1
    fh.to_absolute(cutoff).to_relative(cutoff)  # failure 2


@pytest.mark.skipif(
    not _check_soft_dependencies("pandas>=2.1.0", severity="none"),
    reason="frequency logic requires pandas>=2.1.0",
)
@pytest.mark.parametrize("ts", [True, False])
def test_pandas22_freq_roundtrip(ts):
    """Test that to_absolute and to_relative conversions work with the airline data.

    Failure case in bug #6572.
    """
    y = load_airline()
    if ts:
        y.index = y.index.to_timestamp()

    f = NaiveForecaster(strategy="last")
    f.fit(y)

    fh = ForecastingHorizon([0], is_relative=True)
    fh.to_absolute(f.cutoff)
    fh.to_absolute(f.cutoff).to_relative(f.cutoff)


def test_timestamp_format_to_absolute():
    cutoff = pd.Timestamp("2025-03-02 12:00:00")
    fh = ForecastingHorizon([1, 2, 3], freq="D")
    y_pred_idx = fh.to_absolute_index(cutoff)
    assert "12:00:00" in str(y_pred_idx)
