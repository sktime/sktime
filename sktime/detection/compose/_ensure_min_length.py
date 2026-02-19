"""Compositor to ensure minimum segment length in detection output."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["chpatel"]
__all__ = ["EnsureMinLengthSegments"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class EnsureMinLengthSegments(BaseDetector):
    """Ensure segments from a detector have at least a given minimum length.

    Compositor that wraps a detector and post-processes its output so that
    all segments (episodes) are at least ``min_length`` time steps long.
    Segments shorter than ``min_length`` are merged with adjacent segments.

    Parameters
    ----------
    estimator : sktime detector, i.e., estimator inheriting from BaseDetector
        The detector whose output will be post-processed.
        This is a "blueprint" estimator; state does not change when ``fit`` is called.
    min_length : int, default=2
        Minimum allowed length for any segment. Segments shorter than this
        are merged according to the chosen ``strategy``.
    strategy : str, default="merge_sequential"
        Strategy for merging short segments. One of:

        * ``"merge_sequential"``: iterate over segments in order; if a segment
          is shorter than ``min_length``, merge it with the next segment and
          assign the majority label. Repeat until no short segments remain.
        * ``"shortest_first"``: process segments starting from the shortest.
          For each too-short segment, split it at the midpoint and merge each
          half with the adjacent segment. Re-sort by length after each merge
          and repeat until no short segments remain.

    Attributes
    ----------
    estimator_ : sktime detector, clone of ``estimator``
        Fitted clone of the wrapped detector.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy._zero_cp import ZeroChangePoints
    >>> from sktime.detection.compose._ensure_min_length import (
    ...     EnsureMinLengthSegments,
    ... )
    >>> X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d = EnsureMinLengthSegments(ZeroChangePoints(), min_length=2)
    >>> d.fit(X)
    EnsureMinLengthSegments(...)
    >>> d.predict(X)
    Empty DataFrame
    Columns: [ilocs]
    Index: []
    """

    _tags = {
        "authors": ["patelchaitany"],
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": False,
    }

    def __init__(self, estimator, min_length=2, strategy="merge_sequential"):
        self.estimator = estimator
        self.min_length = min_length
        self.strategy = strategy

        super().__init__()

        allowed_strategies = ("merge_sequential", "shortest_first")
        if strategy not in allowed_strategies:
            raise ValueError(
                f"strategy must be one of {allowed_strategies}, got '{strategy}'."
            )

        tags_to_clone = [
            "task",
            "learning_type",
            "capability:multivariate",
            "capability:missing_values",
        ]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y=None):
        """Fit the wrapped detector to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self : reference to self.
        """
        self.estimator_ = self.estimator.clone()
        self.estimator_.fit(X=X, y=y)
        return self

    def _predict(self, X):
        """Create annotations with minimum segment length enforced.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
        """
        y_pred = self.estimator_.predict(X)
        n = len(X)

        dense = self.sparse_to_dense(y_pred, index=pd.RangeIndex(n))
        if isinstance(dense, pd.DataFrame):
            labels = dense.iloc[:, 0].values.copy()
        else:
            labels = dense.values.copy()

        if self.strategy == "merge_sequential":
            labels = self._merge_sequential(labels, self.min_length)
        else:
            labels = self._merge_shortest_first(labels, self.min_length)

        task = self.get_tag("task")
        if task in ("anomaly_detection", "change_point_detection"):
            return self._labels_to_change_points(labels)
        else:
            return self._labels_to_segments(labels)

    @staticmethod
    def _get_segments(labels):
        """Convert a dense label array into a list of (start, end, label) tuples."""
        segments = []
        n = len(labels)
        if n == 0:
            return segments
        seg_start = 0
        current_label = labels[0]
        for i in range(1, n):
            if labels[i] != current_label:
                segments.append((seg_start, i, current_label))
                seg_start = i
                current_label = labels[i]
        segments.append((seg_start, n, current_label))
        return segments

    @staticmethod
    def _segments_to_labels(segments, n):
        """Convert a list of (start, end, label) tuples back to a dense label array."""
        labels = np.zeros(n, dtype=int)
        for start, end, label in segments:
            labels[start:end] = label
        return labels

    @staticmethod
    def _merge_sequential(labels, min_length):
        """Merge short segments sequentially with the next segment.

        Iterate over segments in order. If a segment is shorter than
        ``min_length``, merge it with the next segment and assign
        the majority label of the two. Repeat until no short segments remain.
        """
        changed = True
        while changed:
            changed = False
            segments = EnsureMinLengthSegments._get_segments(labels)
            new_segments = []
            i = 0
            while i < len(segments):
                start, end, label = segments[i]
                seg_len = end - start
                if seg_len < min_length and i + 1 < len(segments):
                    next_start, next_end, next_label = segments[i + 1]
                    merged_len_current = end - start
                    merged_len_next = next_end - next_start
                    if merged_len_current >= merged_len_next:
                        merged_label = label
                    else:
                        merged_label = next_label
                    new_segments.append((start, next_end, merged_label))
                    i += 2
                    changed = True
                else:
                    new_segments.append((start, end, label))
                    i += 1

            n = len(labels)
            labels = EnsureMinLengthSegments._segments_to_labels(new_segments, n)

        return labels

    @staticmethod
    def _merge_shortest_first(labels, min_length):
        """Merge short segments starting from the shortest.

        Process segments from shortest to longest. For each too-short segment,
        split it at the midpoint and merge each half with the adjacent segment.
        Re-sort by length after each merge and repeat until no short segments
        remain.
        """
        n = len(labels)

        while True:
            segments = EnsureMinLengthSegments._get_segments(labels)
            short = [
                (end - start, idx, start, end, label)
                for idx, (start, end, label) in enumerate(segments)
                if end - start < min_length
            ]
            if not short:
                break

            short.sort()
            seg_len, idx, start, end, label = short[0]

            if len(segments) == 1:
                break

            mid = start + seg_len // 2

            new_segments = []
            for j, (s, e, lbl) in enumerate(segments):
                if j == idx:
                    continue
                if j == idx - 1:
                    new_segments.append((s, mid, lbl))
                elif j == idx + 1:
                    new_segments.append((mid, e, lbl))
                else:
                    new_segments.append((s, e, lbl))

            if idx == 0:
                if new_segments:
                    s0, e0, l0 = new_segments[0]
                    new_segments[0] = (start, e0, l0)
            if idx == len(segments) - 1:
                if new_segments:
                    sl, el, ll = new_segments[-1]
                    new_segments[-1] = (sl, end, ll)

            labels = EnsureMinLengthSegments._segments_to_labels(new_segments, n)

        return labels

    @staticmethod
    def _labels_to_change_points(labels):
        """Convert dense labels to change point indices."""
        change_points = np.where(np.diff(labels) != 0)[0] + 1
        return pd.Series(change_points, dtype="int64")

    @staticmethod
    def _labels_to_segments(labels):
        """Convert dense labels to sparse segment format with IntervalIndex."""
        segments = EnsureMinLengthSegments._get_segments(labels)
        if not segments:
            return BaseDetector._empty_segments()
        starts = [s for s, e, l in segments]
        ends = [e for s, e, l in segments]
        seg_labels = [l for s, e, l in segments]
        index = pd.IntervalIndex.from_arrays(starts, ends, closed="left")
        return pd.Series(seg_labels, index=index, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sktime.detection.dummy._zero_cp import ZeroChangePoints

        params1 = {
            "estimator": ZeroChangePoints(),
            "min_length": 2,
            "strategy": "merge_sequential",
        }
        params2 = {
            "estimator": ZeroChangePoints(),
            "min_length": 3,
            "strategy": "shortest_first",
        }
        return [params1, params2]
