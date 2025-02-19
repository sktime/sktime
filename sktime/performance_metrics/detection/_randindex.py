import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class RandIndex(BaseDetectionMetric):
    r"""Segmentation Rand Index metric.

    The Segmentation Rand Index (SRI) for comparing two segmentations
    is the integrated version of the Rand Index
    for clustering (RI). It measures the similarity between two segmentations of the
    same time series, where each segmentation is a sequence of non-overlapping segments.

    It takes values between 0 and 1, where 1 indicates identical segmentations.

    The segmentation Rand Index (SRI) is obtained from the clustering Rand Index (RI),
    in qualitative terms, by:

    - Interpreting segments as clusters along an index or time axis.

    - Counting pairwise “agreements” (same vs different) along that axis.

    A mathematical definition follows.

    For clarity, we define the clustering Rand Index (RI) first:

    The Rand Index (RI) between two clusterings of n elements is defined as:

    .. math:: RI = (a + b) / (a + b + c + d),

    where:

    - a = #pairs in the same cluster in both segmentations

    - b = #pairs in different clusters in both segmentations

    - c = #pairs in the same cluster in the first segmentation but different in the second

    - d = #pairs in different clusters in the first segmentation but same in the second

    The segmentation Rand Index (SRI) is defined as follows:

    For a time series starting at time index :math:`S \in \mathbb{R}`
    and ending at time index :math:`T \in \mathbb{R}`, let

    :math:`a: [S, T] \rightarrow \mathbb{N}` be the first segmentation,
    and :math:`b: [S, T] \rightarrow \mathbb{N}` be the second segmentation.

    Let :math:`\mathbb{I}(s_1, s_2, t_1, t_2)` be the indicator function that is 1
    iff at least one of the following holds, and 0 otherwise:

    - :math:`s_1 = s_2` and :math:`t_1 = t_2`
    - :math:`s_1 \neq s_2` and :math:`t_1 \neq t_2`

    Then the SRI is defined as:

    .. math:: SRI = \frac{1}{T-S} \int_S^T \int_S^T \mathbb{I}(a(s), a(t), b(s), b(t)) ds dt

    By default, if ``X`` is provided, this metric computes distances in loc-based units
    by looking up the corresponding labels in ``X.index``.
    Otherwise, or if ``use_loc=False``, it uses iloc-based indexing.

    If an existing "label" column is present in y_true/y_pred, those labels are used
    directly for matching; otherwise, segments are considered to be numbered
    consecutively, starting from 0.
    """  # noqa: E501

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "segments",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # Higher Rand Index is better
    }

    def __init__(self, use_loc=True):
        """
        Parameters
        ----------
        use_loc : bool, optional (default=True)

            - If True (and X is provided), segment lengths/overlaps are computed as the
              difference of X.index[end_iloc] - X.index[start_iloc].

            - If False, or X=None, uses iloc-based distances (end_iloc - start_iloc) as before.
        """  # noqa: D205, E501
        self.use_loc = use_loc
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """
        Calculate Rand Index between true and predicted segments.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth segments with 'start', 'end', or an 'ilocs' column (interval or int),
            plus an optional 'label' column.

        y_pred : pd.DataFrame
            Predicted segments with 'start', 'end', or an 'ilocs' column (interval or int),
            plus an optional 'label' column.

        X : pd.DataFrame, optional (default=None)
            If provided (and use_loc=True), loc-based distances are used.

        Returns
        -------
        float
            Rand Index score between 0.0 and 1.0.
        """  # noqa: E501
        # 1) Validate and extract segments (still iloc-based).
        y_true_segments = self._extract_segments(y_true, "y_true")
        y_pred_segments = self._extract_segments(y_pred, "y_pred")

        # 2) Assign unique cluster IDs to each segment (does not overwrite 'label').
        y_true_segments = self._assign_unique_ids(y_true_segments, prefix="true")
        y_pred_segments = self._assign_unique_ids(y_pred_segments, prefix="pred")

        # 3) Determine total_length in loc or iloc units
        if X is not None and self.use_loc:
            # Loc-based => difference of X.index[-1] - X.index[0], if more than one row
            if len(X) > 1:
                total_length = float(X.index[-1] - X.index[0])
            else:
                total_length = len(X)
        else:
            # Iloc-based => max 'end' among true/pred
            max_end_true = max((seg["end"] for seg in y_true_segments), default=0)
            max_end_pred = max((seg["end"] for seg in y_pred_segments), default=0)
            total_length = max(max_end_true, max_end_pred)

        # 4) If there's <2 points in total, the Rand Index is trivially 1.
        if total_length < 2:
            return 1.0

        # 5) Compute same-cluster pairs in y_true, y_pred
        same_cluster_true = sum(
            self._pairs_count(self._compute_length(seg["start"], seg["end"], X))
            for seg in y_true_segments
        )
        same_cluster_pred = sum(
            self._pairs_count(self._compute_length(seg["start"], seg["end"], X))
            for seg in y_pred_segments
        )

        # 6) same-cluster pairs that both segmentations agree on
        same_cluster_both = self._compute_same_cluster_both(
            y_true_segments, y_pred_segments, X
        )

        # 7) total pairs
        total_pairs = self._pairs_count(total_length)

        # 8) Rand Index = (a + d) / total_pairs, with
        #    a = same_cluster_both
        #    d = total_pairs - same_cluster_true - same_cluster_pred + a
        agreements = same_cluster_both + (
            total_pairs - same_cluster_true - same_cluster_pred + same_cluster_both
        )
        rand_index = agreements / total_pairs
        return rand_index

    def _compute_length(self, start_iloc, end_iloc, X):
        """Compute the length of a segment in either loc-based or iloc-based units."""
        if end_iloc <= start_iloc:
            return 0
        length_iloc = end_iloc - start_iloc
        if X is not None and self.use_loc:
            # clamp to avoid out-of-bounds
            start_iloc = max(0, min(start_iloc, len(X) - 1))
            end_iloc = max(0, min(end_iloc, len(X) - 1))
            return float(X.index[end_iloc] - X.index[start_iloc])
        else:
            return float(length_iloc)

    def _compute_same_cluster_both(self, y_true_segments, y_pred_segments, X=None):
        """Compute # of same-cluster pairs (a) in the overlap of segments with the same label."""  # noqa: E501
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
                overlap_length = self._compute_length(overlap_start, overlap_end, X)
                # Compare existing 'label' columns
                if t_seg["label"] == p_seg["label"]:
                    a += self._pairs_count(overlap_length)

            if t_seg["end"] <= p_seg["end"]:
                i += 1
            else:
                j += 1

        return a

    def _pairs_count(self, length):
        """
        Number of unique pairs among 'length' items.

        For discrete or continuous:

        - If length < 2, returns 0.

        - Otherwise, we round length to integer n and compute (n*(n-1))//2
        """  # noqa: D401
        if length < 2:
            return 0
        n = int(round(length))
        return (n * (n - 1)) // 2 if n >= 2 else 0

    def _extract_segments(self, y, var_name):
        """
        Extract segments from the DataFrame.

        1) 'start'/'end' => direct use

        2) A single int column 'ilocs' => interpret each row as [i, i+1)

        3) A single interval column 'ilocs' => read left/right from each Interval

        If a 'label' column is present, that is used directly for identifying clusters.
        Otherwise, the row index i is used as a fallback label.
        """
        segments = []

        # Case 1
        if {"start", "end"}.issubset(y.columns):
            for i, row in y.iterrows():
                seg_start = row["start"]
                seg_end = row["end"]
                seg_label = row["label"] if "label" in y.columns else i
                segments.append(
                    {"start": seg_start, "end": seg_end, "label": seg_label}
                )
            return segments

        # Case 2/3: user-provided 'ilocs'
        if "ilocs" in y.columns:
            col_dtype = y["ilocs"].dtype
            if isinstance(col_dtype, pd.IntervalDtype):
                for i, row in y.iterrows():
                    interval = row["ilocs"]
                    seg_start, seg_end = interval.left, interval.right
                    seg_label = row["label"] if "label" in y.columns else i
                    segments.append(
                        {"start": seg_start, "end": seg_end, "label": seg_label}
                    )
            else:
                # integer => [iloc_val, iloc_val+1)
                for i, row in y.iterrows():
                    iloc_val = row["ilocs"]
                    seg_label = row["label"] if "label" in y.columns else i
                    segments.append(
                        {"start": iloc_val, "end": iloc_val + 1, "label": seg_label}
                    )
            return segments

        raise ValueError(
            f"Expected columns ['start','end'] or 'ilocs' in {var_name}, "
            f"but found columns {list(y.columns)}."
        )

    def _assign_unique_ids(self, segments, prefix):
        """
        Assign a unique 'cluster_id' to each segment (for internal reference),
        but do not overwrite the existing 'label' if it is present.

        Parameters
        ----------
        segments : list of dict
            Each dict has 'start', 'end', and 'label'.

        prefix : str
            E.g. "true" or "pred", to differentiate ground truth vs predicted.

        Returns
        -------
        segments : list of dict
            Same as input, but each dict gains a 'cluster_id' key.
        """  # noqa: D205
        for idx, seg in enumerate(segments):
            # We do NOT overwrite seg['label'] -- we only create a separate ID if needed
            seg["cluster_id"] = f"{prefix}_{seg['label']}_{idx}"
        return segments

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Creates two configurations:

        1) default RandIndex (use_loc=True)

        2) RandIndex(use_loc=False)
        """
        param1 = {}  # default => use_loc=True
        param2 = {"use_loc": False}
        return [param1, param2]
