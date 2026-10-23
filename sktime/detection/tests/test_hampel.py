"""Regression tests for HampelDetector panel support (issue #11615)."""

import pandas as pd
import pytest

from sktime.datatypes import convert_to
from sktime.detection.hampel import HampelDetector
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(HampelDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


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

