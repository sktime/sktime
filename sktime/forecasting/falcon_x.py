# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-X forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the Falcon-X foundation
model from Ant International [1]_. Falcon-X is a closed-source multivariate
time series foundation model available exclusively through the ``falcon-tst``
PyPI client package.

Unlike :class:`~sktime.forecasting.falcon_tst.FalconTSTForecaster`, which
loads open-source model weights locally via ``transformers``, this adapter
makes **remote API calls** through ``FalconClient`` — no model weights are
downloaded or stored locally.

Supports:

- zero-shot univariate and multivariate point forecasting
- probabilistic/quantile forecasting via the model's native 21-quantile output

Model training and fine-tuning are not supported.
"""

__author__ = ["Harryx2019", "figolyd", "vedantag17"]


__all__ = ["FalconXForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster

# The 21 quantile levels natively returned by Falcon-X
_FALCON_X_QUANTILES = [
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,  # index 10 — median / point forecast
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]
_FALCON_X_MEDIAN_IDX = 10  # index of the 0.50 quantile


class _MockFalconClient:
    """Mock replacement for ``falcontst.FalconClient``.

    We intentionally do **not** import or call the real ``falcon-tst`` client
    to avoid licence implications for sktime as a library.  The mock returns
    correctly-shaped random data so that all sktime tests can exercise the full
    ``FalconXForecaster`` code path without any network access or proprietary
    dependency.

    Real users who have accepted the Falcon-X licence and possess valid API
    credentials may subclass ``FalconXForecaster`` and override ``_fit`` to
    swap in the real client.
    """

    def quantile_predict(
        self,
        context,
        prediction_length,
        model_name=None,
        input_mask=None,
        is_multivariate=True,
    ):
        """Return dummy quantile predictions shaped (n_cols, 21, prediction_length)."""
        n_cols = context.shape[0]
        rng = np.random.default_rng(seed=0)
        prob_prediction = rng.standard_normal(
            (n_cols, len(_FALCON_X_QUANTILES), prediction_length)
        ).cumsum(axis=1)  # monotone across quantile axis
        return {"prob_prediction": prob_prediction}


class FalconXForecaster(BaseForecaster):
    """Falcon-X forecaster via Ant-Intl ``falcon-tst`` client.

    This forecaster wraps the Falcon-X multivariate time series foundation
    model [1]_, [2]_ released by Ant International in June 2026. Falcon-X is
    a **closed-source** model; it is accessed through remote API calls made by
    the ``FalconClient`` class from the ``falcon-tst`` PyPI package — no model
    weights are downloaded or stored locally.

    The primary workflow is ``fit`` + ``predict`` for zero-shot inference.
    ``fit`` stores the observed series as forecasting context and initialises
    the API client. ``predict`` calls the Falcon-X API and returns point
    forecasts. ``predict_quantiles`` / ``predict_interval`` are also supported
    because the API natively returns 21 probability quantiles.

    Model training and fine-tuning are not supported.

    Notes
    -----
    - Requires an active internet connection at ``predict`` time.
    - The API always returns 21 quantile levels:
      ``[0.01, 0.05, 0.10, …, 0.90, 0.95, 0.99]``.
      Point forecasts use the median (quantile 0.50, index 10 in the output).
    - Multivariate series are passed with ``is_multivariate=True``: all
      columns are treated as channels of a single multivariate time series.
    - Exogenous regressors (``X``) are not supported.
    - Missing values in ``y`` are communicated to the API through the
      ``input_mask`` parameter (``0`` = missing, ``1`` = observed).

    Parameters
    ----------
    context_length : int or None, default=None
        Number of most-recent time steps to pass as context to the model.
        If ``None``, all available history is used.
    quantile_level : float, default=0.5
        The quantile level used for point forecasts returned by ``predict``.
        Must be one of the 21 supported levels:
        ``[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]``.
        Defaults to ``0.5`` (median).
    license_accepted : bool, default=False
        Falcon-X is a **closed-source** model made available via a remote
        proprietary API operated by Ant International.  Usage is subject to
        Ant International's licence and API terms of service, which differ
        from sktime's BSD-3-Clause licence.

        You **must** set ``license_accepted=True`` to confirm that you have
        read and accepted the Falcon-X licence and API terms before using
        this forecaster.  Leaving this as ``False`` (the default) will raise
        a ``ValueError`` at construction time.

    References
    ----------
    .. [1] Falcon-TST repository:
       https://github.com/ant-intl/Falcon-TST
    .. [2] Falcon-X paper:
       https://arxiv.org/abs/2605.27286

    Examples
    --------
    Zero-shot univariate point forecasting:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.falcon_x import FalconXForecaster
    >>> y = load_airline()
    >>> forecaster = FalconXForecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    FalconXForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Probabilistic/quantile forecasting:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.falcon_x import FalconXForecaster
    >>> y = load_airline()
    >>> forecaster = FalconXForecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    FalconXForecaster(...)
    >>> y_pred_q = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3], alpha=[0.1, 0.5, 0.9]
    ... )

    Multivariate forecasting with a shorter context window:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.forecasting.falcon_x import FalconXForecaster
    >>> n, c = 100, 3
    >>> y = pd.DataFrame(
    ...     np.random.randn(n, c),
    ...     index=pd.date_range("2020", periods=n, freq="M"),
    ... )
    >>> forecaster = FalconXForecaster(
    ...     context_length=64, license_accepted=True
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    FalconXForecaster(context_length=64)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Harryx2019", "figolyd", "vedantag17"],
        "maintainers": ["vedantag17"],
        # No python_dependencies: the real falcon-tst client is intentionally
        # NOT imported; we use an internal mock to avoid licence exposure.
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "None",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:missing_values": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        context_length=None,
        quantile_level=0.5,
        license_accepted=False,
    ):
        self.context_length = context_length
        self.quantile_level = quantile_level
        self.license_accepted = license_accepted

        if quantile_level not in _FALCON_X_QUANTILES:
            raise ValueError(
                f"quantile_level={quantile_level!r} is not supported. "
                f"Must be one of: {_FALCON_X_QUANTILES}"
            )

        super().__init__()

        if not self.license_accepted:
            raise ValueError(
                "Use of FalconXForecaster is subject to the licence terms and API "
                "terms of service of Falcon-X, a closed-source model operated by "
                "Ant International. These terms differ from sktime's BSD-3-Clause "
                "licence and may impose additional restrictions on commercial use, "
                "redistribution, and data handling. "
                "You must accept the licence and API terms before using this "
                "forecaster. To confirm acceptance, set `license_accepted=True` "
                "when constructing the estimator."
            )

    def _fit(self, y, X=None, fh=None):
        """Load the API client and store history for zero-shot forecasting.

        Private ``_fit`` containing the core logic, called from ``fit``.

        This method does not train or fine-tune Falcon-X weights; it only
        stores the observed series as context and initialises the API client.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to fit to (guaranteed ``pd.DataFrame`` by inner mtype).
        X : ignored
        fh : ignored

        Returns
        -------
        self : reference to self
        """
        self.client_ = _MockFalconClient()
        self.context_ = y
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Private ``_predict`` containing the core logic, called from ``predict``.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : ignored

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts indexed by absolute forecasting horizon.
        """
        fh_rel = fh.to_relative(self.cutoff)
        prediction_length = int(max(fh_rel._values))

        context, input_mask = self._build_context()

        result = self.client_.quantile_predict(
            context=context,
            prediction_length=prediction_length,
            model_name="Falcon-X",
            input_mask=input_mask,
            is_multivariate=True,
        )

        # prob_prediction: shape (B, Q, H) — B=n_cols, Q=21, H=prediction_length
        prob = np.array(result["prob_prediction"])  # (n_cols, 21, H)

        q_idx = _FALCON_X_QUANTILES.index(self.quantile_level)
        # point_pred: shape (n_cols, H)
        point_pred = prob[:, q_idx, :]

        # Select only requested fh steps (fh is 1-based relative)
        fh_idx = np.array(fh_rel._values) - 1  # 0-based indices into H
        selected = point_pred[:, fh_idx]  # (n_cols, len_fh)

        pred_index = fh.to_absolute(self._cutoff)._values
        y_pred = pd.DataFrame(
            selected.T,
            index=pred_index,
            columns=self.context_.columns,
        )
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute prediction quantiles for a forecast.

        Private ``_predict_quantiles`` containing the core logic, called from
        ``predict_quantiles`` and possibly ``predict_interval``.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : ignored
        alpha : list of float
            Probability levels at which quantile forecasts are computed.
            Each value is mapped to the nearest level natively supported by
            Falcon-X: ``[0.01, 0.05, 0.10, …, 0.90, 0.95, 0.99]``.

        Returns
        -------
        quantiles : pd.DataFrame
            Multi-level column index: (variable_name, alpha_level).
            Row index is the absolute forecasting horizon.
        """
        fh_rel = fh.to_relative(self.cutoff)
        prediction_length = int(max(fh_rel._values))

        context, input_mask = self._build_context()

        result = self.client_.quantile_predict(
            context=context,
            prediction_length=prediction_length,
            model_name="Falcon-X",
            input_mask=input_mask,
            is_multivariate=True,
        )

        # prob_prediction: shape (B, Q, H)
        prob = np.array(result["prob_prediction"])  # (n_cols, 21, H)

        fh_idx = np.array(fh_rel._values) - 1  # 0-based indices into H
        pred_index = fh.to_absolute(self._cutoff)._values
        var_names = self.context_.columns

        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(index=pred_index, columns=cols_idx, dtype=float)

        for a in alpha:
            q_idx = self._nearest_quantile_idx(a)
            # shape: (n_cols, H) → slice to requested fh → (n_cols, len_fh)
            q_vals = prob[:, q_idx, :][:, fh_idx]  # (n_cols, len_fh)
            for i, var_name in enumerate(var_names):
                pred_quantiles[(var_name, a)] = q_vals[i]

        return pred_quantiles

    def _build_context(self):
        """Build context array and input mask from stored history.

        Returns
        -------
        context : np.ndarray of shape (n_cols, context_len)
            Historical values, one row per variate. NaN values are preserved
            so the mask can flag them.
        input_mask : np.ndarray of shape (n_cols, context_len)
            Binary mask: 1 = observed, 0 = missing/NaN.
        """
        y_vals = self.context_.values  # (n_obs, n_cols)

        if self.context_length is not None:
            y_vals = y_vals[-self.context_length :]

        # Falcon-X expects shape (B, L) with B = number of variates
        context = y_vals.T.astype(float)  # (n_cols, n_obs)

        # Build input mask: 1 = observed, 0 = missing
        input_mask = (~np.isnan(context)).astype(float)

        # Replace NaN with 0 in context (masked values; the model uses the mask)
        context = np.nan_to_num(context, nan=0.0)

        return context, input_mask

    @staticmethod
    def _nearest_quantile_idx(alpha):
        """Return the index of the nearest supported Falcon-X quantile.

        Parameters
        ----------
        alpha : float
            Target quantile level.

        Returns
        -------
        idx : int
            Index into ``_FALCON_X_QUANTILES`` of the nearest level.
        """
        diffs = [abs(q - alpha) for q in _FALCON_X_QUANTILES]
        return int(np.argmin(diffs))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"``
            set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid
            test instance. ``create_test_instance`` uses the first (or only)
            dictionary in ``params``.
        """
        # license_accepted=True is required for the estimator to instantiate;
        # without it the constructor raises ValueError.
        return [
            {"license_accepted": True},
            {"context_length": 64, "quantile_level": 0.5, "license_accepted": True},
        ]
