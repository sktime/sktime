"""Base class for detection metrics."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.datatypes import check_is_scitype, convert_to
from sktime.detection._datatypes._check import (
    _is_points_dtype,
    _is_segments_dtype,
)
from sktime.detection._datatypes._convert import (
    _convert_points_to_segments,
    _convert_segments_to_points,
)
from sktime.performance_metrics.base import BaseMetric


class BaseDetectionMetric(BaseMetric):
    """Base class for defining detection error metrics in sktime."""

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "requires_y_true": True,  # if False, is unsupervised metric
        "lower_is_better": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __call__(self, y_true=None, y_pred=None, X=None):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.
            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.
            Not required for unsupervised metrics.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        return self.evaluate(y_true, y_pred, X)

    def evaluate(self, y_true=None, y_pred=None, X=None):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.

            Not required if unsupervised metric,
            that is, if tag ``requires_y_true`` is False.

            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

            Required if tag ``requires_X`` is True.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, X_inner = self._check_ys(y_true, y_pred, X)

        # pass to inner function
        out = self._evaluate(y_true=y_true_inner, y_pred=y_pred_inner, X=X_inner)

        if not isinstance(out, float):
            out = float(out)
        return out

    def _evaluate(self, y_true, y_pred, X):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.
            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        raise NotImplementedError("Abstract method.")

    def _check_ys(self, y_true, y_pred, X):
        SCITYPES = ["Series"]
        INNER_MTYPES = ["pd.DataFrame"]

        def _coerce_to_df(y, var_name="y", allow_none=False):
            if allow_none and y is None:
                return None
            valid, msg, _ = check_is_scitype(
                y, scitype=SCITYPES, return_metadata=[], var_name=var_name
            )
            if not valid:
                raise TypeError(msg)
            y_inner = convert_to(y, to_type=INNER_MTYPES)
            return y_inner

        allow_none_y_true = not self.get_tag("requires_y_true")
        allow_none_X = not self.get_tag("requires_X")

        # catch the case where y_pred is passed as single positional arg
        if allow_none_y_true and y_pred is None and y_true is not None:
            y_pred = y_true
            y_true = None

        y_true = _coerce_to_df(y_true, var_name="y_true", allow_none=allow_none_y_true)
        y_pred = _coerce_to_df(y_pred, var_name="y_pred")
        X = _coerce_to_df(X, var_name="X", allow_none=allow_none_X)

        # coerce to detection type
        y_true = self._coerce_to_detection_type(y_true, X, allow_none=allow_none_y_true)
        y_pred = self._coerce_to_detection_type(y_pred, X)

        return y_true, y_pred, X

    def _coerce_to_detection_type(self, y, X, allow_none=False):
        """Coerce input to detection type."""
        if allow_none and y is None:
            return None

        detection_type = self.get_tag("scitype:y")
        if X is None:
            len_X = None
        else:
            len_X = len(X)

        if _is_points_dtype(y) and detection_type == "segments":
            y = _convert_points_to_segments(y, len_X=len_X)
        elif _is_segments_dtype(y) and detection_type == "points":
            y = _convert_segments_to_points(y, len_X=len_X)
        return y
