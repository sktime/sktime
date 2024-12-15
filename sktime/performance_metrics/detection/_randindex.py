from collections import Counter

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
            Ground truth segments
        y_pred : pd.DataFrame
            Predicted segments
        X : pd.DataFrame, optional (default=None)
            Not used

        Returns
        -------
        float
            Rand Index score between 0.0 and 1.0
        """
        # Convert DataFrame format to segments list
        ground_truth_segments = [
            (row.Index, row.Index + 1) for row in y_true.itertuples()
        ]
        predicted_segments = [(row.Index, row.Index + 1) for row in y_pred.itertuples()]

        # 1) Determine total length (N)
        total_length = max(
            max(end for _, end in ground_truth_segments),
            max(end for _, end in predicted_segments),
        )

        # 2) Create label arrays: ground_truth_labels[i] = cluster_id of i in ground-truth  # noqa: E501
        #                         predicted_labels[i]    = cluster_id of i in predicted
        ground_truth_labels = [None] * total_length
        predicted_labels = [None] * total_length

        # Assign cluster IDs for ground truth
        # We can just use an incremental ID for each segment
        for cluster_id, (start, end) in enumerate(ground_truth_segments):
            for i in range(start, end):
                ground_truth_labels[i] = cluster_id

        # Assign cluster IDs for predicted
        for cluster_id, (start, end) in enumerate(predicted_segments):
            for i in range(start, end):
                predicted_labels[i] = cluster_id

        # 3) Count the size of each ground-truth cluster and predicted cluster
        gt_cluster_size = Counter(ground_truth_labels)
        pred_cluster_size = Counter(predicted_labels)

        # 4) Count how many points fall into each (gt_cluster, pred_cluster) pair
        intersection_count = Counter()
        for i in range(total_length):
            g = ground_truth_labels[i]
            p = predicted_labels[i]
            intersection_count[(g, p)] += 1

        # 5) Compute the number of same-cluster pairs in ground_truth, predicted, and in both  # noqa: E501
        def pairs_count(n):
            # number of ways to choose 2 out of n
            return (n * (n - 1)) // 2 if n >= 2 else 0

        same_gt = sum(pairs_count(sz) for sz in gt_cluster_size.values())
        same_pred = sum(pairs_count(sz) for sz in pred_cluster_size.values())
        same_both = sum(pairs_count(cnt) for cnt in intersection_count.values())

        # 6) Total number of pairs
        total_pairs = pairs_count(total_length)  # = N*(N-1)/2

        # 7) Rand Index formula
        # Agreements = a + d = [pairs same in both] + [pairs different in both]
        agreements = 2 * same_both + total_pairs - same_gt - same_pred
        rand_index = (
            agreements / total_pairs if total_pairs else 1.0
        )  # handle edge case if N < 2  # noqa: E501

        return rand_index
