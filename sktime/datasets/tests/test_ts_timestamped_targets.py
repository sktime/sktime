"""Regression coverage for timestamped target-label files, issue #3521."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_from_tsfile


@pytest.mark.parametrize(
    "timestamps",
    [
        ("0", "1"),
        ("2024-01-01", "2024-01-02"),
        ("2024-01-01 17:00:00", "2024-01-01 17:10:00"),
    ],
)
@pytest.mark.parametrize("labels", ["targetlabel true", "classlabel true 1.5 -2.5"])
def test_load_timestamped_targets(tmp_path, timestamps, labels):
    """Read continuous targets without treating them as enumerated class labels."""
    first, second = timestamps
    path = tmp_path / "timestamped_targets.ts"
    path.write_text(
        "@problemName TimestampedTargets\n@timestamps true\n"
        f"@univariate false\n@{labels}\n@data\n"
        f"({first},1),({second},2):({first},3),({second},4):1.5\n"
        f"({first},5),({second},6):({first},7),({second},8):-2.5\n"
    )
    X, y = load_from_tsfile(path, y_dtype="float")
    np.testing.assert_array_equal(y, [1.5, -2.5])
    assert X.shape == (2, 2)
    expected_index = pd.Index([0, 1]) if first == "0" else pd.DatetimeIndex(timestamps)
    for row in range(2):
        for column in range(2):
            series = X.iloc[row, column]
            pd.testing.assert_index_equal(series.index, expected_index)
            start = row * 4 + column * 2 + 1
            np.testing.assert_array_equal(series, [start, start + 1])


@pytest.mark.parametrize("timestamp", ["0", "2024-01-01", "2024-01-01 17:00:00"])
def test_timestamped_class_label_validation_is_preserved(tmp_path, timestamp):
    """Unknown classification labels must still fail rather than be accepted."""
    path = tmp_path / "invalid_class.ts"
    path.write_text(
        "@problemName InvalidClass\n@timestamps true\n"
        "@univariate true\n@classlabel true known\n@data\n"
        f"({timestamp},1):unknown\n"
    )
    with pytest.raises(OSError, match="class value.*not valid"):
        load_from_tsfile(path)
