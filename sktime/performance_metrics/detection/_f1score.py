from sktime.performance_metrics.detection._windowed import _BaseWindowedDetectionScore


class WindowedF1Score(_BaseWindowedDetectionScore):
    r"""F1-score for event detection, using a margin-based match criterion.

    This score computes the harmonic mean of windowed precision and recall for
    point-event detection.

    If :math:`P` denotes windowed precision and :math:`R` denotes windowed recall,
    the metric is

    .. math::
        \mathrm{WindowedF1Score} = \frac{2PR}{P + R}.

    By default, matching is iloc based. If ``use_loc=True`` and ``X`` is provided,
    windowing uses the corresponding values from ``X.index`` instead.

    Parameters
    ----------
    margin : int, optional (default=0)
        Symmetric matching margin applied on both sides of a true event when
        ``margin_backward`` and ``margin_forward`` are not provided.
    margin_backward : int or timedelta-like, optional (default=None)
        Allowed backward distance from a true event to a predicted event.
        If None, uses ``margin``.
    margin_forward : int or timedelta-like, optional (default=None)
        Allowed forward distance from a true event to a predicted event.
        If None, uses ``margin``.
    use_loc : bool, optional (default=False)
        If True and ``X`` is provided, compute windows in loc units using ``X.index``.
    """

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute F1 score under the margin-based breakpoint matching logic.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth breakpoints in "points" format. Must have column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted breakpoints in "points" format. Must have column 'ilocs'.
        X : pd.DataFrame, optional (default=None)
            Unused here, but part of the signature.

        Returns
        -------
        float
            F1 score, i.e., 2 * precision * recall / (precision + recall).
        """
        matched_count, true_count, pred_count = self._get_match_counts(
            y_true, y_pred, X=X
        )

        # Handle edge cases
        if true_count == 0 and pred_count == 0:
            return 1.0  # No breakpoints to detect, so consider it perfect by convention
        if true_count == 0:
            return 0.0  # ground truth is empty but predictions exist => precision = 0 => F1=0  # noqa: E501
        if pred_count == 0:
            return 0.0  # predictions are empty but ground truth exists => recall = 0 => F1=0  # noqa: E501

        # Compute precision and recall
        precision = matched_count / pred_count
        recall = matched_count / true_count

        # Compute F1
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
