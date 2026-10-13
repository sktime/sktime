import pandas as pd
import numpy as np
from sktime.detection.supervised import BaseSupervisedDetector


class NaivePretrainWindowDetector(BaseSupervisedDetector):
    """todo: write! We are only dealing with change points first, no segments.
    Maybe it might be confusing to the user that X and y are ordered in the opposite way than in the rest of sktime, but it is more intuitive to have the "training data" as y and the "test data" as X, as for usual supervised learning
    """
    def __init__(self, detection_threshold: float | None = None, window_length=10):
        self.window_length = window_length
        self.in_window_mean = None
        self._in_window_counts = 0
        self.out_window_mean = None
        self.baseline_mean = None
        self._baseline_count = None
        if detection_threshold is not None and detection_threshold < 0:
            raise ValueError("detection_threshold must be non-negative")
        self.detection_threshold = detection_threshold
        self._out_window_counts = 0
        self._is_fitted = False
        self._is_vectorized = False
        super().__init__()

    @staticmethod
    def windows_before_events(X: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
        parts = []
        for inst, X_inst in X.groupby(level="instances", sort=False):
            ilocs = y.loc[y["instances"] == inst, "ilocs"].to_numpy()
            for ev in ilocs:
                parts.append(X_inst.loc[pd.IndexSlice[:, ev - w:ev - 1], :])
        return pd.concat(parts)

    def _pretrain(self, X: pd.DataFrame, y: pd.Series):
        in_window_df = self.windows_before_events(X, y, self.window_length)
        out_window_df = X.loc[~X.index.isin(in_window_df.index)]

        self.in_window_mean = float(in_window_df["value"].mean())
        self.out_window_mean = float(out_window_df["value"].mean())
        self._in_window_counts = len(in_window_df)
        self._out_window_counts = len(out_window_df)
        # todo: should we be able to call .update directly after pretrain? If so then this needs to stay True otherwise we can get rid of this line!
        self._is_fitted = True
        self._state = "fitted"  #"pretrained"
        return self

    def _predict(self, X: pd.DataFrame):
        if self.detection_threshold is None:
            if self.in_window_mean is None or self.out_window_mean is None:
                raise ValueError(
                    "pretrain must be called before predict if detection_threshold is not set"
                )
            detection_threshold = self.in_window_mean - self.out_window_mean
        else:
            detection_threshold = self.detection_threshold

        trailing_mean = (
            X.groupby(level="instances")["value"]
             .tail(self.window_length)
             .groupby(level="instances")
             .mean()
        )
        detected = trailing_mean[
            (trailing_mean - self.baseline_mean).abs() > detection_threshold
        ]

        last_iloc = (
            X.reset_index()
             .groupby("instances")["timepoints"]
             .max()
             .loc[detected.index]
        )
        return pd.DataFrame({
            "instances": detected.index,
            "ilocs": last_iloc.values,
        })

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        if self._state == "new":
            if y is None:
                raise ValueError(
                    "y is required when fit is called before pretrain"
                )
            self._pretrain(X, y)
        # reset previous baseline state
        self.baseline_mean = None
        self._baseline_count = None
        self._update(y=y, X=X)
        self._state = "fitted"
        return self

    def _update(self, y: pd.Series, X: pd.DataFrame):
        new_mean = float(X["value"].mean())
        new_count = len(X)

        if self.baseline_mean is None:
            self.baseline_mean = new_mean
            self._baseline_count = new_count
        else:
            self.baseline_mean = (
                self.baseline_mean * self._baseline_count + new_mean * new_count
            ) / (self._baseline_count + new_count)
            self._baseline_count = self._baseline_count + new_count

        return self



