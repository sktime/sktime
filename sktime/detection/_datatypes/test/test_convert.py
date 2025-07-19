"""Tests for detection output converters."""

import pandas as pd
import pytest

from sktime.detection._datatypes._convert import (
    _convert_points_to_segments,
    _convert_segments_to_points,
)
from sktime.detection._datatypes._examples import (
    _get_example_points_2,
    _get_example_points_3,
    _get_example_segments_2,
    _get_example_segments_3,
)
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection._datatypes"),
    reason="Test only runs when module changed",
)
def test_convert_points_to_segments():
    """Test _convert_points_to_segments on an example."""
    points_df = _get_example_points_2()
    segments_df_no_lab = _convert_points_to_segments(points_df, len_X=10)
    segments_df_w_lab = _convert_points_to_segments(
        points_df, len_X=10, include_labels=True
    )
    expected_segments_df_w_lab = _get_example_segments_2()
    expected_segments_df_no_lab = expected_segments_df_w_lab.drop(columns=["labels"])

    pd.testing.assert_frame_equal(segments_df_no_lab, expected_segments_df_no_lab)
    pd.testing.assert_frame_equal(segments_df_w_lab, expected_segments_df_w_lab)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection._datatypes"),
    reason="Test only runs when module changed",
)
def test_convert_segments_to_points():
    """Test _convert_points_to_segments on an example."""
    seg_df = _get_example_segments_3()
    points_df_expected = _get_example_points_3()
    points_df = _convert_segments_to_points(seg_df, len_X=10)

    pd.testing.assert_frame_equal(points_df, points_df_expected)
