"""Windowed precision metrics for event detection."""

from sktime.performance_metrics.detection._windowed import _BaseWindowedDetectionScore


class WindowedPrecision(_BaseWindowedDetectionScore):
    r"""Windowed precision (positive predictive value) for event detection.

    This score computes precision for point-event detection using one-to-one matching
    within a window around each true event.

    Denoting by :math:`M` the number of matched predicted/true event pairs and by
    :math:`|\hat{Y}|` the number of predicted events, the metric is

    .. math::
        \mathrm{WindowedPrecision} = \frac{M}{|\hat{Y}|}.

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import WindowedPrecision
    >>> y_true = pd.DataFrame({"ilocs": [1, 4, 10]})
    >>> y_pred = pd.DataFrame({"ilocs": [1, 5, 8, 13]})
    >>> WindowedPrecision(margin=1)(y_true, y_pred)
    0.5
    """

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute windowed precision."""
        matched_count, true_count, pred_count = self._get_match_counts(
            y_true, y_pred, X=X
        )

        if true_count == 0 and pred_count == 0:
            return 1.0
        if pred_count == 0:
            return 0.0

        return matched_count / pred_count
