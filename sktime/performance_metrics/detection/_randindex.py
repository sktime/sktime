import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class RandIndex(BaseDetectionMetric):
    """Rand Index metric for comparing event detection results."""

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "segments",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # Higher Rand Index is better
    }

    def _evaluate(self, y_true, y_pred, X=None):
        """Calculate Rand Index between true and predicted segments.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth segments with 'start', 'end', or an 'ilocs' column (interval or int),
            plus an optional 'label' column.
        y_pred : pd.DataFrame
            Predicted segments with 'start', 'end', or an 'ilocs' column (interval or int),
            plus an optional 'label' column.
        X : pd.DataFrame, optional (default=None)
            If provided, used for index alignment.

        Returns
        -------
        float
            Rand Index score between 0.0 and 1.0.
        """  # noqa: E501
        # Validate and extract segments
        y_true_segments = self._extract_segments(y_true, "y_true")
        y_pred_segments = self._extract_segments(y_pred, "y_pred")

        # Assign unique cluster IDs to each segment
        y_true_segments = self._assign_unique_ids(y_true_segments, prefix="true")
        y_pred_segments = self._assign_unique_ids(y_pred_segments, prefix="pred")

        # Determine total length N
        if X is not None:
            # Use the length of X (or its index) for alignment
            total_length = len(X)
        else:
            # Use maximum 'end' value from y_true and y_pred
            max_end_true = max((seg["end"] for seg in y_true_segments), default=0)
            max_end_pred = max((seg["end"] for seg in y_pred_segments), default=0)
            total_length = max(max_end_true, max_end_pred)

        # If there's <2 points in total, the Rand Index is trivially 1
        if total_length < 2:
            return 1.0

        # Compute # of same-cluster pairs in ground truth and predicted
        same_cluster_true = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_true_segments
        )
        same_cluster_pred = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_pred_segments
        )

        # Compute # of same-cluster pairs that both segmentations agree on
        same_cluster_both = self._compute_same_cluster_both(
            y_true_segments, y_pred_segments
        )  # noqa: E501

        # Total # of pairs among total_length
        total_pairs = self._pairs_count(total_length)

        # Rand Index = (a + d) / total_pairs,
        #   where a = same_cluster_both,
        #         d = total_pairs - same_cluster_true - same_cluster_pred + a
        agreements = same_cluster_both + (
            total_pairs - same_cluster_true - same_cluster_pred + same_cluster_both
        )
        rand_index = agreements / total_pairs
        return rand_index

    def _extract_segments(self, y, var_name):
        """Extract segments from the DataFrame.

        Handles three cases:
        1) Columns "start" and "end" (direct use)
        2) A single integer column "ilocs" => interpret each row as [i, i+1)
        3) A single interval column "ilocs" => extract left/right from each Interval

        Parameters
        ----------
        y : pd.DataFrame
            Must contain either:
              - "start", "end" (and optionally "label"), or
              - "ilocs" as integer, or
              - "ilocs" as interval (pandas.Interval).
        var_name : str
            Name for error messages, e.g. "y_true" or "y_pred".

        Returns
        -------
        list of dict
            Each dict includes "start", "end", "label".
        """
        segments = []

        # Case 1: user-provided 'start' and 'end'
        if {"start", "end"}.issubset(y.columns):
            for i, row in y.iterrows():
                seg_start = row["start"]
                seg_end = row["end"]
                seg_label = row["label"] if "label" in y.columns else i
                segments.append(
                    {"start": seg_start, "end": seg_end, "label": seg_label}
                )  # noqa: E501
            return segments

        # Case 2 or 3: user-provided 'ilocs'
        if "ilocs" in y.columns:
            col_dtype = y["ilocs"].dtype
            if pd.api.types.is_interval_dtype(col_dtype):
                # Each row is a pandas.Interval => [left, right)
                for i, row in y.iterrows():
                    interval = row["ilocs"]
                    seg_start, seg_end = interval.left, interval.right
                    seg_label = row["label"] if "label" in y.columns else i
                    segments.append(
                        {"start": seg_start, "end": seg_end, "label": seg_label}
                    )  # noqa: E501
            else:
                # Assume each 'ilocs' is an integer => [i, i+1)
                for i, row in y.iterrows():
                    iloc_val = row["ilocs"]
                    seg_start, seg_end = iloc_val, iloc_val + 1
                    seg_label = row["label"] if "label" in y.columns else i
                    segments.append(
                        {"start": seg_start, "end": seg_end, "label": seg_label}
                    )  # noqa: E501
            return segments

        # If neither approach applies, raise an error
        raise ValueError(
            f"Expected columns ['start','end'] or 'ilocs' in {var_name}, "
            f"but found columns {list(y.columns)}."
        )

    def _assign_unique_ids(self, segments, prefix):
        """Assign unique cluster IDs to each segment.

        Parameters
        ----------
        segments : list of dict
            Each dict has 'start', 'end', and 'label'.
        prefix : str
            Prefix to differentiate between ground-truth vs predicted segments.

        Returns
        -------
        list of dict
            Each dict includes a unique 'cluster_id' along with 'start', 'end', 'label'.
        """
        for idx, seg in enumerate(segments):
            seg["cluster_id"] = f"{prefix}_{seg['label']}_{idx}"
        return segments

    def _compute_same_cluster_both(self, y_true_segments, y_pred_segments):
        """Compute the number of pairs a, where both segmentations agree on same cluster.

        Overlaps of segments with same "label" => pairs_count(overlap_length).

        Parameters
        ----------
        y_true_segments : list of dict
            "start", "end", "label", "cluster_id".
        y_pred_segments : list of dict
            "start", "end", "label", "cluster_id".

        Returns
        -------
        int
            # of pairs that belong to the same cluster in both y_true and y_pred.
        """  # noqa: E501
        a = 0
        # Sort segments by their start
        y_true_sorted = sorted(y_true_segments, key=lambda x: x["start"])
        y_pred_sorted = sorted(y_pred_segments, key=lambda x: x["start"])

        i, j = 0, 0
        while i < len(y_true_sorted) and j < len(y_pred_sorted):
            t_seg = y_true_sorted[i]
            p_seg = y_pred_sorted[j]

            # Overlap interval
            overlap_start = max(t_seg["start"], p_seg["start"])
            overlap_end = min(t_seg["end"], p_seg["end"])

            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                if t_seg["label"] == p_seg["label"]:
                    a += self._pairs_count(overlap_length)

            # Move to the next segment in whichever ends first
            if t_seg["end"] <= p_seg["end"]:
                i += 1
            else:
                j += 1

        return a

    def _pairs_count(self, n):
        """Number of unique pairs among n items."""  # noqa: D401
        return (n * (n - 1)) // 2 if n >= 2 else 0
