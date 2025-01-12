import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class RandIndex(BaseDetectionMetric):
    """Rand Index metric for comparing event detection results.

    Optionally computes the Rand Index in loc-based units (index labels of X)
    if X is provided and use_loc=True. Otherwise uses iloc-based intervals.

    Parameters
    ----------
        use_loc : bool, optional (default=True)
            If True, and X is provided, interpret 'start'/'end' in the DataFrame
            as *labels in X.index* rather than 0-based positions.
            They will be converted to integer positions internally before
            computing the Rand Index. If False, or if X=None, the code
            uses 'start'/'end' (or 'ilocs') as raw integers/positions as before.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "segments",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # Higher Rand Index is better
    }

    def __init__(self, use_loc=True):
        self.use_loc = use_loc
        super().__init__()

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
            If provided and use_loc=True, 'start'/'end' are interpreted as labels in X.index.

        Returns
        -------
        float
            Rand Index score between 0.0 and 1.0.
        """  # noqa: E501
        # 1) Extract segments as a list of {"start", "end", "label"}
        y_true_segments = self._extract_segments(y_true, "y_true")
        y_pred_segments = self._extract_segments(y_pred, "y_pred")

        # 2) If user wants loc-based and X is given, convert from label -> integer positions  # noqa: E501
        if self.use_loc and X is not None:
            # Build a dictionary to map label -> integer position
            # e.g., if X.index = [10, 11, 15, 42, 50], then index_map[10] = 0, index_map[11] = 1, etc.  # noqa: E501
            index_map = {label: i for i, label in enumerate(X.index)}
            # Convert each segment's start/end to integer positions
            y_true_segments = self._loc_to_iloc_segments(y_true_segments, index_map)
            y_pred_segments = self._loc_to_iloc_segments(y_pred_segments, index_map)

        # 3) Assign unique cluster IDs to each segment
        y_true_segments = self._assign_unique_ids(y_true_segments, prefix="true")
        y_pred_segments = self._assign_unique_ids(y_pred_segments, prefix="pred")

        # 4) Determine total length N
        if X is not None:
            # Use the length of X (or its index) for alignment
            total_length = len(X)
        else:
            # Use maximum 'end' value from y_true and y_pred
            max_end_true = max((seg["end"] for seg in y_true_segments), default=0)
            max_end_pred = max((seg["end"] for seg in y_pred_segments), default=0)
            total_length = max(max_end_true, max_end_pred)

        # 5) Edge case: if <2 points in total, Rand Index is trivially 1
        if total_length < 2:
            return 1.0

        # 6) same-cluster pairs in ground truth and predicted
        same_cluster_true = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_true_segments
        )
        same_cluster_pred = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_pred_segments
        )

        # 7) same-cluster pairs in both
        same_cluster_both = self._compute_same_cluster_both(
            y_true_segments, y_pred_segments
        )  # noqa: E501

        # 8) total pairs in [0..N)
        total_pairs = self._pairs_count(total_length)

        # Rand Index = (a + d) / total_pairs
        # where a = same_cluster_both
        #       d = total_pairs - same_cluster_true - same_cluster_pred + a
        agreements = same_cluster_both + (
            total_pairs - same_cluster_true - same_cluster_pred + same_cluster_both
        )
        rand_index = agreements / total_pairs
        return rand_index

    def _loc_to_iloc_segments(self, segments, index_map):
        """Convert 'start'/'end' labels to integer positions using index_map.

        Parameters
        ----------
        segments : list of dict
            Each dict has 'start', 'end', 'label'.
            'start'/'end' are assumed to be labels in X.index that exist in index_map.
        index_map : dict
            Maps label -> integer position in X.

        Returns
        -------
        list of dict
            The same segments, but with 'start'/'end' replaced by integer positions.
        """
        new_segments = []
        for seg in segments:
            # If the user actually stored integers but we do have index_map, we should check  # noqa: E501
            # whether seg["start"] in index_map. If it’s not, just keep as is or raise error.  # noqa: E501, RUF003
            start_val = seg["start"]
            end_val = seg["end"]
            try:
                start_iloc = index_map[start_val]
                end_iloc = index_map[end_val]
            except KeyError:
                # if the user passed loc-based start/end that doesn't exist in index_map
                raise ValueError(
                    f"Segment {seg} references label(s) not found in X.index: "
                    f"{start_val} or {end_val}"
                )
            # We assume inclusive-exclusive or exclusive-exclusive?
            # The original code was exclusive at 'end', so be consistent
            new_segments.append(
                {"start": start_iloc, "end": end_iloc, "label": seg["label"]}
            )
        return new_segments

    def _extract_segments(self, y, var_name):
        """Extract segments from the DataFrame.

        1) Columns "start" and "end" => direct use
        2) A single integer column "ilocs" => interpret each row as [i, i+1)
        3) A single interval column "ilocs" => extract left/right from each Interval
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
        """Assign unique cluster IDs to each segment."""
        for idx, seg in enumerate(segments):
            seg["cluster_id"] = f"{prefix}_{seg['label']}_{idx}"
        return segments

    def _compute_same_cluster_both(self, y_true_segments, y_pred_segments):
        """Compute # of pairs a where both segmentations agree on same label (overlaps)."""  # noqa: E501
        a = 0
        y_true_sorted = sorted(y_true_segments, key=lambda x: x["start"])
        y_pred_sorted = sorted(y_pred_segments, key=lambda x: x["start"])

        i, j = 0, 0
        while i < len(y_true_sorted) and j < len(y_pred_sorted):
            t_seg = y_true_sorted[i]
            p_seg = y_pred_sorted[j]

            overlap_start = max(t_seg["start"], p_seg["start"])
            overlap_end = min(t_seg["end"], p_seg["end"])

            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                if t_seg["label"] == p_seg["label"]:
                    a += self._pairs_count(overlap_length)

            if t_seg["end"] <= p_seg["end"]:
                i += 1
            else:
                j += 1

        return a

    def _pairs_count(self, n):
        """Number of unique pairs among n items."""  # noqa: D401
        return (n * (n - 1)) // 2 if n >= 2 else 0

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Creates two configurations:
        1) default RandIndex (use_loc=True)
        2) RandIndex(use_loc=False)
        """
        param1 = {}  # relies on default use_loc=True
        param2 = {"use_loc": False}
        return [param1, param2]
