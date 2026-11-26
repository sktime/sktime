# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Diebold-Mariano test analyzer for comparing forecaster accuracy."""

__all__ = ["DieboldMarianoTest"]

import numpy as np
import pandas as pd

from sktime.base import BaseObject
from sktime.utils.dependencies import _check_estimator_deps


def _to_1d_array(obj, name):
    """Convert Series, 1-column DataFrame, or array-like to a 1D numpy float array."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(
                f"{name} must be 1-dimensional, got DataFrame with {obj.shape[1]} "
                "columns."
            )
        arr = obj.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy(dtype=float)
    else:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-dimensional, got array with shape {arr.shape}."
            )
    return arr


class DieboldMarianoTest(BaseObject):
    """Diebold-Mariano test for equal predictive accuracy of two forecasters.

    Stateless analyzer that compares two competing forecasts, ``y_pred_a`` and
    ``y_pred_b``, against observed ground truth ``y_true`` using the
    Diebold-Mariano (1995) test with optional Harvey-Leybourne-Newbold (1997)
    small-sample adjustment. Delegates computation to
    ``statsmodels.tsa.stattools.diebold_mariano_test``.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon used to compute the default number of HAC lags and
        the Harvey-Leybourne-Newbold (1997) small-sample adjustment. Must be >= 1.
    harvey_adj : bool, default=True
        Whether to apply the Harvey-Leybourne-Newbold (1997) small-sample
        correction (rescaled statistic compared against Student's t with
        ``nobs - 1`` degrees of freedom).
    criterion : str or callable, default="mse"
        Loss function used to score each forecast. Passed to
        ``statsmodels.tsa.stattools.diebold_mariano_test`` (``"mse"``,
        ``"mad"``/``"mae"``, ``"mape"``, ``"poly"``, or a callable
        ``loss(y_true, y_pred)``).
    lags : int or None, default=None
        Number of lags for the Newey-West HAC variance estimator. If ``None``,
        defaults to ``max(horizon - 1, ceil(nobs ** (1/3)))``.
    power : float, default=2
        Exponent for ``criterion="poly"``. Ignored for other criteria.
    """

    _tags = {
        "object_type": "benchmark_analyzer",
        "python_dependencies": ["statsmodels"],
        "property:analyzer_type": "pairwise",
    }

    def __init__(
        self,
        horizon=1,
        harvey_adj=True,
        criterion="mse",
        lags=None,
        power=2,
    ):
        self.horizon = horizon
        self.harvey_adj = harvey_adj
        self.criterion = criterion
        self.lags = lags
        self.power = power

        super().__init__()
        _check_estimator_deps(self)

    def evaluate(self, y_true, y_pred_a, y_pred_b):
        """Run the Diebold-Mariano test on two competing forecast series.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame (1 column), or 1D array-like
            Observed ground truth values.
        y_pred_a : pd.Series, pd.DataFrame (1 column), or 1D array-like
            Forecasted values from the first model.
        y_pred_b : pd.Series, pd.DataFrame (1 column), or 1D array-like
            Forecasted values from the second model.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns
            ``["statistic", "p_val", "lags", "harvey_adj_factor", "nobs"]``.
        """
        from statsmodels.tsa.stattools import diebold_mariano_test as _sm_dm_test

        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError(f"horizon must be an integer >= 1, got {self.horizon}.")

        y_arr = _to_1d_array(y_true, "y_true")
        fa_arr = _to_1d_array(y_pred_a, "y_pred_a")
        fb_arr = _to_1d_array(y_pred_b, "y_pred_b")

        if not (len(y_arr) == len(fa_arr) == len(fb_arr)):
            raise ValueError(
                "y_true, y_pred_a, and y_pred_b must have the same length, got "
                f"{len(y_arr)}, {len(fa_arr)}, {len(fb_arr)}."
            )
        if len(y_arr) < 2:
            raise ValueError(
                f"At least 2 observations are required, got {len(y_arr)}."
            )

        res = _sm_dm_test(
            y_arr,
            fa_arr,
            fb_arr,
            lags=self.lags,
            criterion=self.criterion,
            power=self.power,
            harvey_adj=self.harvey_adj,
            horizon=int(self.horizon),
        )

        return pd.DataFrame(
            [
                {
                    "statistic": float(res.statistic),
                    "p_val": float(res.pvalue),
                    "lags": int(res.lags),
                    "harvey_adj_factor": (
                        float(res.harvey_adj_factor)
                        if res.harvey_adj_factor is not None
                        else np.nan
                    ),
                    "nobs": len(y_arr),
                }
            ]
        )
