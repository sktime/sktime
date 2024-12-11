"""Base class for detection metrics."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.datatypes import check_is_scitype, convert_to
from sktime.performance_metrics.base import BaseMetric


class BaseDetectionMetric(BaseMetric):
    """Base class for defining detection error metrics in sktime."""

    _tags = {
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "lower_is_better": True,
    }

    def __call__(self, y_true, y_pred, X=None):
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

    def evaluate(self, y_true, y_pred, X=None):
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

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

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

    def _evaluate(self, y_true, y_pred, X=None):
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

        y_true = _coerce_to_df(y_true, var_name="y_true")
        y_pred = _coerce_to_df(y_pred, var_name="y_pred")
        X = _coerce_to_df(X, var_name="X", allow_none=True)

        return y_true, y_pred, X
