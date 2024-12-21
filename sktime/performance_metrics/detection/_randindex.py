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
            Ground truth segments with 'start', 'end', and optional 'label' columns.
        y_pred : pd.DataFrame
            Predicted segments with 'start', 'end', and optional 'label' columns.
        X : pd.DataFrame, optional (default=None)
            If provided, used for index alignment.

        Returns
        -------
        float
            Rand Index score between 0.0 and 1.0.
        """
        # Validate and extract segments
        y_true_segments = self._extract_segments(y_true, "y_true")
        y_pred_segments = self._extract_segments(y_pred, "y_pred")

        # Assign unique cluster IDs to each segment
        y_true_segments = self._assign_unique_ids(y_true_segments, prefix="true")
        y_pred_segments = self._assign_unique_ids(y_pred_segments, prefix="pred")

        # Determine total length N
        if X is not None:
            # Use index from X for alignment
            total_length = len(X)
        else:
            # Use maximum 'end' value from y_true and y_pred
            max_end_true = (
                max(seg["end"] for seg in y_true_segments) if y_true_segments else 0
            )  # noqa: E501
            max_end_pred = (
                max(seg["end"] for seg in y_pred_segments) if y_pred_segments else 0
            )  # noqa: E501
            total_length = max(max_end_true, max_end_pred)

        if total_length < 2:
            return 1.0  # Perfect agreement by definition

        # Compute same_cluster_true and same_cluster_pred
        same_cluster_true = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_true_segments
        )  # noqa: E501
        same_cluster_pred = sum(
            self._pairs_count(seg["end"] - seg["start"]) for seg in y_pred_segments
        )  # noqa: E501

        # Compute same_cluster_both (a)
        same_cluster_both = self._compute_same_cluster_both(
            y_true_segments, y_pred_segments
        )  # noqa: E501

        # Total number of pairs
        total_pairs = self._pairs_count(total_length)

        # Compute Rand Index
        agreements = same_cluster_both + (
            total_pairs - same_cluster_true - same_cluster_pred + same_cluster_both
        )  # noqa: E501
        rand_index = agreements / total_pairs

        return rand_index

    def _extract_segments(self, y, var_name):
        """Extract segments from the DataFrame.

        Parameters
        ----------
        y : pd.DataFrame
            DataFrame containing segment information.
        var_name : str
            Variable name for error messages.

        Returns
        -------
        list of dict
            Each dict represents a segment with 'start', 'end', and 'label'.
        """
        seg_ix = y.set_index("ilocs").index
        seg_dict = {"start": seg_ix.left, "end": seg_ix.right}
        if "label" in y.columns:
            seg_dict["label"] = y["label"]
        else:
            seg_dict["label"] = range(len(y))

        return seg_dict

    def _assign_unique_ids(self, segments, prefix):
        """Assign unique cluster IDs to each segment.

        Parameters
        ----------
        segments : list of dict
            Each dict represents a segment with 'start', 'end', and 'label'.
        prefix : str
            Prefix to differentiate between true and predicted clusters.

        Returns
        -------
        list of dict
            Each dict includes a unique 'cluster_id' along with 'start', 'end', and 'label'.
        """  # noqa: E501
        for idx, seg in enumerate(segments):
            seg["cluster_id"] = f"{prefix}_{seg['label']}_{idx}"
        return segments

    def _compute_same_cluster_both(self, y_true_segments, y_pred_segments):
        """Compute the number of agreeing pairs (a) where both true and predicted clusters agree.

        Parameters
        ----------
        y_true_segments : list of dict
            Each dict includes 'cluster_id', 'start', 'end', and 'label' for true segments.
        y_pred_segments : list of dict
            Each dict includes 'cluster_id', 'start', 'end', and 'label' for predicted segments.

        Returns
        -------
        int
            Number of agreeing pairs.
        """  # noqa: E501
        a = 0
        # Create sorted lists based on start times
        y_true_sorted = sorted(y_true_segments, key=lambda x: x["start"])
        y_pred_sorted = sorted(y_pred_segments, key=lambda x: x["start"])

        i, j = 0, 0
        while i < len(y_true_sorted) and j < len(y_pred_sorted):
            true_seg = y_true_sorted[i]
            pred_seg = y_pred_sorted[j]

            # Find overlap
            overlap_start = max(true_seg["start"], pred_seg["start"])
            overlap_end = min(true_seg["end"], pred_seg["end"])

            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                if true_seg["label"] == pred_seg["label"]:
                    a += self._pairs_count(overlap_length)

            # Move to the next segment
            if true_seg["end"] <= pred_seg["end"]:
                i += 1
            else:
                j += 1

        return a

    def _pairs_count(self, n):
        """Calculate the number of ways to choose 2 items from n items.

        Parameters
        ----------
        n : int
            Number of items.

        Returns
        -------
        int
            Number of unique pairs.
        """
        return (n * (n - 1)) // 2 if n >= 2 else 0
