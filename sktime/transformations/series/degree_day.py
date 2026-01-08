"""Degree-day feature transformer.

This module implements a transformer that computes Heating Degree Days (HDD) and
Cooling Degree Days (CDD) from daily temperature inputs.

Degree days convert temperature into a simple proxy for heating/cooling demand by
measuring how far a day's mean temperature is from a base (balance-point)
temperature.

Definitions (daily):
    tmean = (tmax + tmin) / 2
    HDD   = max(0, base_temp - tmean)
    CDD   = max(0, tmean - base_temp)
"""

import pandas as pd

from sktime.transformations.base import BaseTransformer


class DegreeDayFeatures(BaseTransformer):
    """Compute degree-day features (HDD/CDD) from daily temperatures.

    Supports two input modes:
    - If no column names are provided: 1 column is treated as mean temperature (tmean);
      2+ columns use the first two columns as (tmax, tmin).
    - If column names are provided, the transformer uses those columns.

    Parameters
    ----------
    base_temp : float, default=65.0
        Base (balance-point) temperature in the same units as the input.
    tmax_col, tmin_col : str or None, default=None
        Column names for daily max/min temperature. If both provided, uses them.
    tmean_col : str or None, default=None
        Column name for mean temperature. If provided and present, uses it.
    return_tmean : bool, default=True
        If True, include `tmean` in the output.
    strict : bool, default=False
        If True, raises when tmin > tmax; if False, auto-swaps those rows.
    keep_original_columns : bool, default=False
        If True, appends features to X. If False, returns only features.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:missing_values": False,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        base_temp: float = 65.0,
        tmax_col: str | None = None,
        tmin_col: str | None = None,
        tmean_col: str | None = None,
        return_tmean: bool = True,
        strict: bool = False,
        keep_original_columns: bool = False,
    ):
        """Initialize the transformer.

        Parameters
        ----------
        tmax_col, tmin_col : str or None
            If both provided, use them as max/min temperature columns.
        tmean_col : str or None
            If provided (and present), use it as mean temperature column.
        strict : bool
            If using tmax/tmin: if True, raise when tmin > tmax; if False, auto-swap.

        Notes
        -----
        If no column names are provided:
        - if X has 1 column, it is treated as tmean
        - if X has >= 2 columns, first two are treated as (tmax, tmin)
        """
        self.base_temp = float(base_temp)
        self.tmax_col = tmax_col
        self.tmin_col = tmin_col
        self.tmean_col = tmean_col
        self.return_tmean = bool(return_tmean)
        self.strict = bool(strict)
        self.keep_original_columns = bool(keep_original_columns)
        super().__init__()

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self._validate_input(X)

        # Choose temperature source
        mode = self._infer_mode(X)

        if mode == "tmean":
            tmean = self._extract_tmean(X)
        else:
            tmax, tmin = self._extract_tmax_tmin(X)
            tmax, tmin = self._handle_inverted_min_max(X, tmax, tmin)
            tmean = (tmax + tmin) / 2.0

        hdd = (self.base_temp - tmean).clip(lower=0)
        cdd = (tmean - self.base_temp).clip(lower=0)

        feats = pd.DataFrame(index=X.index)
        if self.return_tmean:
            feats["tmean"] = tmean
        feats["hdd"] = hdd
        feats["cdd"] = cdd

        return X.join(feats) if self.keep_original_columns else feats

    # ---- helpers ----

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        return X

    def _infer_mode(self, X: pd.DataFrame) -> str:
        # Highest priority: explicit tmean_col if present
        if self.tmean_col is not None and self.tmean_col in X.columns:
            return "tmean"

        # Next: explicit tmax/tmin if provided
        if self.tmax_col is not None or self.tmin_col is not None:
            if not (self.tmax_col in X.columns and self.tmin_col in X.columns):
                raise ValueError(
                    f"Missing required column(s): '{self.tmax_col}', '{self.tmin_col}'"
                )
            return "tmax_tmin"

        # Auto mode: infer from shape
        if X.shape[1] == 1:
            return "tmean"
        if X.shape[1] >= 2:
            return "tmax_tmin"

        raise ValueError("X must have at least 1 column.")

    def _extract_tmean(self, X: pd.DataFrame) -> pd.Series:
        if self.tmean_col is not None and self.tmean_col in X.columns:
            s = X[self.tmean_col]
        else:
            # auto: take the single column
            s = X.iloc[:, 0]

        tmean = pd.to_numeric(s, errors="coerce")
        if tmean.isna().any():
            raise ValueError(
                "Non-numeric or missing values found in temperature column."
            )
        return tmean

    def _extract_tmax_tmin(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if self.tmax_col is not None and self.tmin_col is not None:
            smax = X[self.tmax_col]
            smin = X[self.tmin_col]
        else:
            # auto: take first two columns
            smax = X.iloc[:, 0]
            smin = X.iloc[:, 1]

        tmax = pd.to_numeric(smax, errors="coerce")
        tmin = pd.to_numeric(smin, errors="coerce")

        if tmax.isna().any() or tmin.isna().any():
            raise ValueError(
                "Non-numeric or missing values found in temperature columns."
            )
        return tmax, tmin

    def _handle_inverted_min_max(
        self, X: pd.DataFrame, tmax: pd.Series, tmin: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        inverted = tmin > tmax
        if not inverted.any():
            return tmax, tmin

        if self.strict:
            raise ValueError(
                "Found rows where tmin > tmax. Set strict=False to auto-swap."
            )

        tmax2 = tmax.copy()
        tmin2 = tmin.copy()
        tmax2[inverted] = tmin[inverted]
        tmin2[inverted] = tmax[inverted]
        return tmax2, tmin2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return parameter settings for sktime's estimator checks."""
        return [
            # Auto mode (works with estimator-generated data)
            {"base_temp": 65.0},
            # Another auto-mode variant (still estimator-safe)
            {"base_temp": 60.0, "return_tmean": False},
            # Keep this if you want; it is estimator-safe because it only uses tmean_col
            # if the column actually exists; otherwise it falls back to auto-mode.
            {"base_temp": 65.0, "tmean_col": "tmean"},
        ]
