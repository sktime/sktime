# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kinematic transformers."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class KinematicFeatures(BaseTransformer):
    r"""Kinematic feature transformer - velocity, acceleration, curvature.

    Takes a discrete N-dimensional space curve, N>=1, and computes
    a selection of kinematic features.

    For noisy time series, is strongly recommended to pipeline this with
    ``KalmanFilterTransformerPK`` or ``KalmanFilterTransformerFP`` (prior),
    or other smoothing or trajectory fitting transformers,
    as this transformer does not carry out its own smoothing.

    For min/max/quantiles of velocity etc, pipeline with ``SummaryTransformer`` (post).

    For a time series input :math:`x(t)`, observed at discrete times,
    this transformer computes (when selected) discretized versions of:

    * ``"v"`` - vector of velocity: :math:`\vec{v}(t) := \Delta x(t)`
    * ``"v_abs"`` - absolute velocity: :math:`v(t) := \left| \Delta x(t) \right|`
    * ``"a"`` - vector of velocity: :math:`\vec{a}(t) := \Delta \Delta x(t)`
    * ``"a_abs"`` - absolute velocity: :math:`a(t) := \left| \Delta \Delta x(t) \right|`
    * ``"curv"`` - curvature: :math:`c(t) := \frac{\sqrt{v(t)^2 a(t)^2 - \left\langle
      \vec{v}(t), \vec{a}(t)\right\rangle^2}}{v(t)^3}`

    where :math:`\Delta` denotes first finite differences, that is,
    :math:`\Delta z(t) = z(t) - z(t-1)` for any discrete time series :math:`z(t)`.

    Note: this estimator currently ignores non-equidistant location index,
    and considers only the integer location index.

    Parameters
    ----------
    features : str or list of str, optional, default=["v_abs", "a_abs", "c_abs"]
        list of features to compute, possible features:

        * "v" - vector of velocity
        * "v_abs" - absolute velocity
        * "a" - vector of acceleration
        * "a_abs" - absolute acceleration
        * "curv" - curvature

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.transformations.series.kinematic import KinematicFeatures

    >>> traj3d = pd.DataFrame(columns=["x", "y", "z"])
    >>> traj3d["x"] = pd.Series(np.sin(np.arange(200)/100))
    >>> traj3d["y"] = pd.Series(np.cos(np.arange(200)/100))
    >>> traj3d["z"] = pd.Series(np.arange(200)/100)

    >>> t = KinematicFeatures()
    >>> Xt = t.fit_transform(traj3d)
    """

    _tags = {
        "authors": ["fkiraly"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "capability:categorical_in_X": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, features=None, remember_data="last"):
        self.features = features
        self.remember_data = remember_data
        
        if features is None:
            self._features = ["v_abs", "a_abs", "curv"]
        elif isinstance(features, str):
            self._features = [features]
        else:
            self._features = features

        # Add a safety check for the new parameter
        if self.remember_data not in ["last", "all", "none"]:
            raise ValueError(
                'remember_data must be one of "last", "all", or "none", '
                f'but found: {self.remember_data}'
            )

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored, present for interface compliance

        Returns
        -------
        transformed version of X
        """
        features = self._features
        res = pd.DataFrame()

        # --- HISTORY MEMORY LOGIC START ---
        # 1. Retrieve historical data if it was saved in a previous run
        X_history = getattr(self, "_X_history", None)
        
        # 2. Prepend history to X to calculate accurate kinematics across chunk borders
        if X_history is not None and not X_history.empty and self.remember_data != "none":
            X_calc = pd.concat([X_history, X])
            history_len = len(X_history)
        else:
            X_calc = X.copy()
            history_len = 0

        # 3. Save the current data as history for the NEXT time transform is called
        if self.remember_data == "all":
            self._X_history = X_calc.copy()
        elif self.remember_data == "last":
            # We only need a maximum of 2 historical points to calculate acceleration/jerk
            self._X_history = X_calc.iloc[-2:].copy()
        else:
            self._X_history = None
        # --- HISTORY MEMORY LOGIC END ---

        def prepend_cols(df, prefix):
            df.columns = [f"{prefix}__{col}" for col in df.columns]
            return df

        def absq_rows(df, col="absq"):
            """Compute DataFrame with one col, absolute value square of rows of df."""
            abs_frame = df**2
            abs_frame = abs_frame.agg(["sum"], axis=1)
            abs_frame.columns = [col]
            return abs_frame

        def abs_rows(df, col="abs"):
            """Compute DataFrame with single column, absolute value of rows of df."""
            return absq_rows(df, col=col) ** 0.5

        def feature_query(queries):
            """Boolean, whether any of the features in queries is being asked for."""
            if queries is str:
                return queries in features
            else:
                return any([x in features for x in queries])

        # NOTE: Replaced X.diff() with X_calc.diff() so math utilizes history
        if feature_query(["v", "v_abs", "curv"]):
            v_frame = X_calc.diff()
            v_frame = prepend_cols(v_frame, "v")
            if feature_query(["v"]):
                res = pd.concat([res, v_frame], axis=1)
            if feature_query(["v_abs"]):
                vabs_frame = abs_rows(v_frame, "v_abs")
                vabs_frame.iloc[0] = np.nan
                res = pd.concat([res, vabs_frame], axis=1)

        if feature_query(["a", "a_abs", "curv"]):
            a_frame = X_calc.diff().diff()
            a_frame = prepend_cols(a_frame, "a")
            if feature_query(["a"]):
                res = pd.concat([res, a_frame], axis=1)
            if feature_query(["a_abs"]):
                aabs_frame = abs_rows(a_frame, "a_abs")
                aabs_frame.iloc[0] = np.nan
                aabs_frame.iloc[1] = np.nan
                res = pd.concat([res, aabs_frame], axis=1)

        if feature_query(["curv"]):
            vsq_frame = absq_rows(v_frame)
            curv_frame = vsq_frame * absq_rows(a_frame)
            curv_arr = curv_frame.values
            cross_term = (v_frame.values * a_frame.values).sum(axis=1) ** 2
            cross_term = cross_term.reshape(-1, 1)
            curv_arr = (curv_arr - cross_term) / (vsq_frame.values**3)
            curv_arr = np.abs(curv_arr) ** 0.5
            # NOTE: Index updated to X_calc to match arrays
            curv_frame = pd.DataFrame(curv_arr, columns=["curv"], index=X_calc.index) 
            res = pd.concat([res, curv_frame], axis=1)

        # --- FINAL CLEANUP START ---
        # Slice off the historical rows we prepended so the output length matches the input X perfectly
        if history_len > 0:
            res = res.iloc[history_len:]
        # --- FINAL CLEANUP END ---

        return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {"features": ["v", "a"]}
        return [params1, params2]
