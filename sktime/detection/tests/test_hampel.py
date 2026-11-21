"""Tests for HampelDetector."""

import pandas as pd
import pytest

from sktime.datatypes import convert_to
from sktime.detection.hampel import HampelDetector
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(HampelDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def test_hampel_single_series():
    """Test HampelDetector on univariate single series."""
    X = pd.Series([1.0, 1.0, 10.0, 1.0, 1.0])
    detector = HampelDetector(window_length=3)
    res = detector.fit_predict(X)

    assert isinstance(res, pd.DataFrame)
    assert "ilocs" in res.columns
    assert list(res["ilocs"]) == [2]

    scores = detector.transform_scores(X)
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (5, 1)
    assert scores.iloc[2, 0] > 10.0


def test_hampel_reproducer_issue_11615():
    """Test the exact reproducer from issue #11615."""
    index = pd.MultiIndex.from_product([["a", "b"], range(5)])
    combined = pd.DataFrame({"value": [1, 1, 10, 1, 1] * 2}, index=index)

    detector = HampelDetector(window_length=3)
    res = detector.fit_predict(combined)

    assert isinstance(res, pd.DataFrame)
    assert "ilocs" in res.columns
    assert list(res["ilocs"]) == [2, 2]
    assert list(res.index.get_level_values(0)) == ["a", "b"]

    scores = detector.transform_scores(combined)
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (10, 1)
    assert scores.index.equals(combined.index)
    assert scores.loc[("a", 2), "scores"] > 10.0
    assert scores.loc[("b", 2), "scores"] > 10.0


@pytest.mark.parametrize("mtype", ["pd-multiindex", "numpy3D", "df-list"])
@pytest.mark.parametrize("method", ["predict", "transform_scores"])
@pytest.mark.parametrize("case", ["issue", "unequal", "constant"])
def test_hampel_panel(mtype, method, case):
    """Panel results equal independent single-series results, including indices."""
    if case == "unequal" and mtype == "numpy3D":
        pytest.skip("numpy3D requires equal length series")

    values = {
        "issue": [[1, 1, 10, 1, 1], [1, 1, 10, 1, 1]],
        "unequal": [[1, 1, 10, 1, 1], [100, 100, 100, 100, 100, 100, 100]],
        "constant": [[1] * 5, [100] * 5],
    }[case]
    series = [pd.DataFrame({"value": v}) for v in values]
    # Non-sorted instance names also check reconstruction order.
    X = pd.concat(series, keys=pd.Index(["b", "a"], name="instance"))
    X.index = X.index.set_names(["instance", "time"])
    panel = convert_to(X, to_type=mtype, as_scitype="Panel")
    detector = HampelDetector(window_length=3).fit(panel)
    actual = getattr(detector, method)(panel)

    expected = []
    # single series worked before, test against this case
    for frame in series:
        single = HampelDetector(window_length=3).fit(frame)
        expected.append(getattr(single, method)(frame))
    keys = X.index.droplevel(-1).unique()
    if mtype != "pd-multiindex":
        keys = pd.RangeIndex(2, name="instances")
    expected = pd.concat(expected, keys=keys)
    if method == "transform_scores" and mtype == "pd-multiindex":
        expected.index = expected.index.set_names(X.index.names)
    elif method == "transform_scores":
        expected.index = expected.index.set_names(["instances", "timepoints"])
    pd.testing.assert_frame_equal(actual, expected)

    if method == "predict":
        assert actual["ilocs"].tolist() == (
            [2, 2] if case == "issue" else [2] if case == "unequal" else []
        )
