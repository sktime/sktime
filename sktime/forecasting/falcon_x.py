# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-X forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the Falcon-X foundation
model from Ant International [1]_. Falcon-X is a closed-source multivariate
time series foundation model accessed through a remote HTTP API.

sktime does **not** ship or depend on any Falcon-X client library.
The forecaster communicates with the Falcon-X API directly via plain
``requests.post`` calls, avoiding any licence implications from vendoring
or depending on the ``falcon-tst`` package.

Supports:

- zero-shot univariate and multivariate point forecasting
- probabilistic/quantile forecasting via the model's native 21-quantile output

Model training and fine-tuning are not supported. Calling :meth:`fit` only
stores the observed series as context for zero-shot prediction.
"""

__author__ = ["Harryx2019", "figolyd", "vedantag17"]

__all__ = ["FalconXForecaster"]

import math

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster

# Default Falcon Studio prediction endpoint
_DEFAULT_PREDICT_URL = (
    "https://falconstudio-pre.antglobal.com/falconstudio/api/v1/openapi/predict"
)

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


def _replace_nan(value):
    """Recursively replace NaN floats with None for JSON serialization."""
    if isinstance(value, list):
        return [_replace_nan(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _post_predict(endpoint, payload, timeout):
    """Send a prediction request to the Falcon-X API.

    This function makes a plain ``requests.post`` call to the Falcon-X
    endpoint.  sktime does **not** import or depend on the ``falcon-tst``
    package — this avoids any licence implications.

    Parameters
    ----------
    endpoint : str
        Full URL of the Falcon Studio prediction endpoint.
    payload : dict
        JSON-serialisable prediction payload.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    dict
        Parsed response with key ``"prob_prediction"``.

    Raises
    ------
    RuntimeError
        If the API returns a non-200 response code or unexpected format.
    """
    import requests

    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Falcon-X API response format: {data}")

    code = data.get("code")
    message = data.get("message", "")
    if code != 200:
        raise RuntimeError(f"Falcon-X API error: code={code}, message={message}")

    return {"prob_prediction": data.get("data")}


def _mock_predict(context, prediction_length):
    """Return random quantile predictions for offline testing.

    Used when ``endpoint="mock"`` is passed to ``FalconXForecaster``.
    Analogous to ``FalconTSTForecaster`` using random weights when
    ``model_path=None`` — allows CI to exercise the full code path
    without network access.

    Parameters
    ----------
    context : np.ndarray of shape (n_channels, context_len)
    prediction_length : int

    Returns
    -------
    dict with key ``"prob_prediction"`` containing an ndarray of shape
    ``(n_channels, 21, prediction_length)``.
    """
    n_channels = context.shape[0]
    rng = np.random.default_rng(seed=0)
    prob_prediction = rng.standard_normal(
        (n_channels, len(_FALCON_X_QUANTILES), prediction_length)
    ).cumsum(axis=1)  # monotone across quantile axis
    return {"prob_prediction": prob_prediction}


class FalconXForecaster(BaseForecaster):
    """Falcon-X forecaster — zero-shot via remote HTTP API.

    This forecaster wraps the Falcon-X multivariate time series foundation
    model [1]_, [2]_ released by Ant International in June 2026. Falcon-X is
    a **closed-source** model; it is accessed through plain HTTP ``POST``
    requests to Ant International's hosted inference endpoint — no model
    weights are downloaded or stored locally, and sktime does **not** ship
    or depend on any Falcon-X client library.

    The primary workflow is ``fit`` + ``predict`` for zero-shot inference.
    ``fit`` stores the observed series as forecasting context. ``predict``
    calls the Falcon-X API and returns point forecasts.
    ``predict_quantiles`` / ``predict_interval`` are also supported because
    the API natively returns 21 probability quantiles.

    Model training and fine-tuning are not supported.

    Notes
    -----
    - Requires the ``requests`` package (``pip install requests``).
    - Requires an active internet connection at ``predict`` time when using
      a real endpoint.
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
    endpoint : str or None, default=None
        Custom API endpoint URL. If ``None``, the default Falcon Studio
        endpoint is used. If ``"mock"``, a built-in mock that returns random
        predictions is used instead of the real API — useful for tests or
        local experimentation without network access.
    timeout : float, default=30.0
        Request timeout in seconds for the API call.

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

    Mock mode for offline testing or local experimentation
    (random predictions, no network access required):

    >>> from sktime.forecasting.falcon_x import FalconXForecaster
    >>> forecaster = FalconXForecaster(
    ...     endpoint="mock", license_accepted=True
    ... )  # doctest: +SKIP

    Multivariate forecasting with a shorter context window:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.forecasting.falcon_x import FalconXForecaster
    >>> n, c = 100, 3
    >>> y = pd.DataFrame(
    ...     np.random.randn(n, c),
    ...     index=pd.date_range("2020", periods=n, freq="ME"),
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
        "python_dependencies": ["requests"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
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
        endpoint=None,
        timeout=30.0,
    ):
        self.context_length = context_length
        self.quantile_level = quantile_level
        self.license_accepted = license_accepted
        self.endpoint = endpoint
        self.timeout = timeout

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
        """Store history for zero-shot forecasting.

        private _fit containing the core logic, called from fit

        This method does not train or fine-tune Falcon-X weights.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self.context_ = y
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast.

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype"
            tag. Point predictions.
        """
        fh_rel = fh.to_relative(self.cutoff)
        prediction_length = int(max(fh_rel._values))

        context, input_mask = self._build_context()

        result = self._call_api(context, input_mask, prediction_length)

        # prob_prediction: shape (B, Q, H) — B=n_cols, Q=21, H=prediction_length
        prob = np.array(result["prob_prediction"])  # (n_cols, 21, H)

        q_idx = _FALCON_X_QUANTILES.index(self.quantile_level)
        point_pred = prob[:, q_idx, :]  # (n_cols, H)

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

        private _predict_quantiles containing the core logic, called from
        predict_quantiles and possibly predict_interval.

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

        result = self._call_api(context, input_mask, prediction_length)

        prob = np.array(result["prob_prediction"])  # (n_cols, 21, H)

        fh_idx = np.array(fh_rel._values) - 1
        pred_index = fh.to_absolute(self._cutoff)._values
        var_names = self.context_.columns

        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(index=pred_index, columns=cols_idx, dtype=float)

        for a in alpha:
            q_idx = self._nearest_quantile_idx(a)
            q_vals = prob[:, q_idx, :][:, fh_idx]  # (n_cols, len_fh)
            for i, var_name in enumerate(var_names):
                pred_quantiles[(var_name, a)] = q_vals[i]

        return pred_quantiles

    def _call_api(self, context, input_mask, prediction_length):
        """Dispatch to real API or mock depending on ``self.endpoint``.

        Parameters
        ----------
        context : np.ndarray of shape (n_cols, context_len)
        input_mask : np.ndarray of shape (n_cols, context_len)
        prediction_length : int

        Returns
        -------
        dict with key ``"prob_prediction"``.
        """
        if self.endpoint == "mock":
            return _mock_predict(context, prediction_length)

        endpoint = self.endpoint if self.endpoint is not None else _DEFAULT_PREDICT_URL

        context_list = _replace_nan(context.tolist())
        mask_list = _replace_nan(input_mask.tolist())

        payload = {
            "context": context_list,
            "prediction_length": prediction_length,
            "model_name": "Falcon-X",
            "input_mask": mask_list,
            "is_multivariate": True,
        }

        return _post_predict(endpoint, payload, self.timeout)

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
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set. There are currently no reserved values for
            forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class. Each dict is a
            parameter set to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid
            test instance. ``create_test_instance`` uses the first (or only)
            dictionary in ``params``.
        """
        # endpoint="mock" returns random predictions without network access,
        # analogous to FalconTSTForecaster(model_path=None) using random
        # weights — allows CI to exercise the full code path offline.
        params1 = {"license_accepted": True, "endpoint": "mock"}
        params2 = {
            "context_length": 64,
            "quantile_level": 0.5,
            "license_accepted": True,
            "endpoint": "mock",
        }

        return [params1, params2]
