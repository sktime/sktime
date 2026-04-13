"""Compositors to enforce a minimum segment length in detector outputs."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["EnsureMinLengthSegments"]

import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.registry import scitype


def _check_strategy(strategy):
    """Validate the smoothing strategy."""
    valid_strategies = {"greedy"}

    if strategy not in valid_strategies:
        raise ValueError(
            f"`strategy` must be one of {sorted(valid_strategies)}, "
            f"but found {strategy!r}."
        )

    return strategy


class EnsureMinLengthSegments(BaseDetector):
    """Enforce a minimum segment length on detector outputs.

    This compositor post-processes the output of a segmentation or change point
    detector. It smooths adjacent segments until all remaining segments have length
    at least ``min_length``, unless the full input series is shorter than
    ``min_length``.

    Parameters
    ----------
    detector : sktime detector, descendant of ``BaseDetector``
        Wrapped detector whose outputs are smoothed.
    min_length : int, default=1
        Minimum allowed segment length, measured in iloc positions.
    strategy : {"greedy"}, default="greedy"
        Smoothing strategy to apply.
        ``"greedy"`` scans from left to right, merging any too-short segment into the
        following segment. If the final segment is too short, it is merged into the
        previous segment.

    Attributes
    ----------
    detector_ : sktime detector
        Clone of ``detector`` fitted during ``fit``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.compose import EnsureMinLengthSegments
    >>> from sktime.detection.skchange_cp import MovingWindow
    >>> X = pd.DataFrame({"value": [0, 0, 1, 1, 0, 0, 1, 1]})
    >>> detector = EnsureMinLengthSegments(
    ...     detector=MovingWindow(change_score="mean", bandwidth=2),
    ...     min_length=2,
    ... )
    >>> detector.fit(X)
    EnsureMinLengthSegments(...)
    >>> detector.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "authors": "sktime developers",
        "tests:core": True,
    }

    def __init__(self, detector, min_length=1, strategy="greedy"):
        self.detector = detector
        self.min_length = min_length
        self.strategy = strategy
        self.detector_ = None

        super().__init__()

        self._check_detector(detector)
        self._check_min_length(min_length)
        self.strategy = _check_strategy(strategy)
        self.detector_ = detector.clone()

        tags_to_clone = [
            "learning_type",
            "task",
            "capability:multivariate",
            "capability:missing_values",
            "capability:update",
            "capability:variable_identification",
            "distribution_type",
            "X_inner_mtype",
        ]
        self.clone_tags(detector, tag_names=tags_to_clone)
        self.set_tags(**{"fit_is_empty": False})

    @staticmethod
    def _check_detector(detector):
        """Validate the wrapped detector."""
        if scitype(detector, raise_on_unknown=False) != "detector":
            raise TypeError(
                "`detector` must be an sktime detector, "
                f"but found {type(detector)!r}."
            )

        task = detector.get_tag("task")
        valid_tasks = {"segmentation", "change_point_detection"}

        if task not in valid_tasks:
            raise ValueError(
                "`EnsureMinLengthSegments` only supports segmentation and change "
                f"point detectors, but found task={task!r}."
            )

    @staticmethod
    def _check_min_length(min_length):
        """Validate ``min_length``."""
        if not isinstance(min_length, int):
            raise TypeError(
                f"`min_length` must be an integer, but found {type(min_length)!r}."
            )
        if min_length < 1:
            raise ValueError(
                f"`min_length` must be at least 1, but found {min_length}."
            )

    def _fit(self, X, y=None):
        """Fit the wrapped detector."""
        self.detector_.fit(X=X, y=y)
        return self

    def _predict(self, X):
        """Predict events after enforcing the minimum segment length."""
        segments = self.detector_.predict_segments(X)
        smoothed_segments = self._smooth_segments(segments)

        if self.get_tag("task") == "segmentation":
            return smoothed_segments

        if len(smoothed_segments) == 0:
            return pd.DataFrame({"ilocs": pd.Series(dtype="int64")})

        change_points = self.segments_to_change_points(smoothed_segments)
        leftmost = smoothed_segments.iloc[0]["ilocs"].left
        change_points = change_points[change_points != leftmost]
        return pd.DataFrame({"ilocs": change_points}).reset_index(drop=True)

    def _predict_scores(self, X):
        """Delegate sparse prediction scores to the wrapped detector."""
        return self.detector_.predict_scores(X)

    def _transform_scores(self, X):
        """Delegate dense transformation scores to the wrapped detector."""
        return self.detector_.transform_scores(X)

    def _update(self, X, y=None):
        """Update the wrapped detector."""
        self.detector_.update(X=X, y=y)
        return self

    def _get_fitted_params(self):
        """Get fitted parameters from the wrapped detector."""
        return self.detector_.get_fitted_params()

    def _smooth_segments(self, segments):
        """Smooth a contiguous segment partition."""
        if len(segments) <= 1 or self.min_length <= 1:
            return segments.reset_index(drop=True)

        keep_columns = ["ilocs"]
        if "labels" in segments.columns:
            keep_columns.append("labels")

        segments = segments.loc[:, keep_columns].reset_index(drop=True)
        self._validate_segments(segments)

        while len(segments) > 1:
            lengths = segments["ilocs"].map(self._segment_length)
            short_segments = lengths < self.min_length

            if not short_segments.any():
                break

            short_idx = short_segments.idxmax()

            if short_idx == len(segments) - 1:
                merge_idx = short_idx - 1
                merged = self._merge_rows(
                    left=segments.iloc[merge_idx],
                    right=segments.iloc[short_idx],
                    prefer="left",
                )
                segments = pd.concat(
                    [
                        segments.iloc[:merge_idx],
                        pd.DataFrame([merged]),
                        segments.iloc[short_idx + 1 :],
                    ],
                    ignore_index=True,
                )
            else:
                merged = self._merge_rows(
                    left=segments.iloc[short_idx],
                    right=segments.iloc[short_idx + 1],
                    prefer="right",
                )
                segments = pd.concat(
                    [
                        segments.iloc[:short_idx],
                        pd.DataFrame([merged]),
                        segments.iloc[short_idx + 2 :],
                    ],
                    ignore_index=True,
                )

        return segments.reset_index(drop=True)

    @staticmethod
    def _validate_segments(segments):
        """Validate that segments form a contiguous partition."""
        ilocs = pd.IntervalIndex(segments["ilocs"])

        if len(ilocs) == 0:
            return

        if not ilocs.is_non_overlapping_monotonic:
            raise ValueError(
                "Wrapped detector must return non-overlapping, monotonic segments."
            )

        left_bounds = ilocs.left.to_numpy()
        right_bounds = ilocs.right.to_numpy()

        if (left_bounds[1:] != right_bounds[:-1]).any():
            raise ValueError(
                "Wrapped detector must return a contiguous segment partition."
            )

    @staticmethod
    def _segment_length(interval):
        """Return the length of an iloc interval."""
        return interval.right - interval.left

    def _merge_rows(self, left, right, prefer):
        """Merge two adjacent rows into a single segment."""
        left_interval = left["ilocs"]
        right_interval = right["ilocs"]

        merged = {
            "ilocs": pd.Interval(
                left_interval.left,
                right_interval.right,
                closed="left",
            )
        }

        if "labels" in left.index and "labels" in right.index:
            merged["labels"] = self._merged_label(
                left_label=left["labels"],
                left_length=self._segment_length(left_interval),
                right_label=right["labels"],
                right_length=self._segment_length(right_interval),
                prefer=prefer,
            )

        return merged

    @staticmethod
    def _merged_label(left_label, left_length, right_label, right_length, prefer):
        """Return the merged label using a length-majority rule."""
        if left_length > right_length:
            return left_label
        if right_length > left_length:
            return right_label
        if prefer == "left":
            return left_label
        return right_label

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection.dummy import ZeroChangePoints, ZeroSegments

        params1 = {"detector": ZeroSegments(), "min_length": 2}
        params2 = {"detector": ZeroChangePoints(), "min_length": 2}

        return [params1, params2]
