# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implements T0 forecaster from The Forecasting Company."""

# This product includes software developed at The Forecasting Company.
# Copyright 2026 The Forecasting Company, Inc.

__all__ = ["T0Forecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton

# minimal architecture used when model_path is None: a small, randomly-initialized
# T0 model built locally (no checkpoint download), for testing / offline use
_RANDOM_MODEL_CONFIG = {
    "embed_dim": 32,
    "num_layers": 1,
    "num_heads": 2,
    "mlp_hidden_dim": 64,
    "patch_size": 8,
    "group_every_n": 1,
    "dropout": 0.0,
    "quantile_levels": [0.1, 0.5, 0.9],
}


class T0Forecaster(BaseForecaster):
    """Interface to the T0 zero-shot forecaster by The Forecasting Company.

    T0 is a pretrained time series foundation model for zero-shot forecasting,
    released by The Forecasting Company [1]_, [2]_. The model is loaded from a
    pretrained checkpoint and applied without task-specific training: the context
    is stored in ``fit`` and forecasts are produced zero-shot in ``predict``.

    T0 natively supports known-future covariates (exogenous data), passed to the
    underlying model via its ``future_covariates`` argument. These covariates must
    span the context window and the forecast horizon, so exogenous data must be
    provided both in ``fit`` (past values) and ``predict`` (future values).

    Capabilities and behaviour:

    * **Multivariate** - a multi-column ``y`` is forecast column-by-column, each as
      an independent series (``capability:multivariate=True``).
    * **Probabilistic** - T0 returns quantiles natively, so ``predict_quantiles``,
      ``predict_interval`` and (by default) ``predict_var`` are available
      (``capability:pred_int=True``); ``predict`` returns the median (0.5 quantile).
    * **Exogenous** - known-future covariates are conditioned on, but must cover
      every step ``1..max(fh)``, so a non-contiguous ``fh`` requires contiguous
      ``X`` (``capability:non_contiguous_X=False``).
    * **Missing values** - ``NaN`` entries in the context are treated as missing
      (``capability:missing_values=True``).
    * **In-sample** - not supported; ``fh`` must be strictly in the future
      (``capability:insample=False``).

    Parameters
    ----------
    model_path : str or None, default="theforecastingcompany/t0-alpha"
        Path to the T0 HuggingFace model checkpoint. The default checkpoint is a
        gated model on the HuggingFace Hub, so using it requires accepting the
        vendor's license (see ``license_accepted``) and authenticating with a
        HuggingFace token. If ``None``, a small randomly-initialized T0 model is
        built locally instead of downloading any checkpoint - this produces
        untrained (meaningless) forecasts and is intended only for testing or
        offline use where the gated checkpoint is unavailable.
    device : str or None, default=None
        Device for inference, e.g., "cpu", "cuda", or "mps". If None, uses "cuda"
        when a CUDA device is available, otherwise "cpu".
    context_length : int or None, default=None
        Maximum context length for inference. If None, the full context is used.
    random_state : int or None, optional, default=None
        Random seed for reproducibility. T0's inference is deterministic, so this
        does not change the forecast; it is accepted for interface consistency.
    license_accepted : bool, optional, default=False
        Whether the user has read and accepted the license terms of the model at
        ``model_path``. The default checkpoint is gated and licensed by The
        Forecasting Company, so ``license_accepted`` must be set to ``True`` to
        load a real checkpoint; otherwise ``fit`` raises. Ignored when
        ``model_path`` is ``None`` (the local random model needs no license).
    ignore_deps : bool, optional, default=False
        If True, dependency checks are skipped.

    Attributes
    ----------
    model_ : t0.T0Forecaster
        The underlying T0 model used for forecasting.

    References
    ----------
    .. [1] https://github.com/theforecastingcompany/tfc-t0
    .. [2] https://huggingface.co/theforecastingcompany/t0-alpha

    Examples
    --------
    Univariate point forecast (zero-shot):

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.t0 import T0Forecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>> forecaster = T0Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Probabilistic forecast (T0 returns quantiles natively):

    >>> y_quantiles = forecaster.predict_quantiles(
    ...     fh=[1, 2, 3], alpha=[0.1, 0.5, 0.9]
    ... )  # doctest: +SKIP
    >>> y_interval = forecaster.predict_interval(
    ...     fh=[1, 2, 3], coverage=0.9
    ... )  # doctest: +SKIP

    Forecast with known-future exogenous data (``X`` in both fit and predict):

    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> y_tr, y_te, X_tr, X_te = temporal_train_test_split(y, X, test_size=3)
    >>> forecaster = T0Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_tr, X=X_tr)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3], X=X_te)  # doctest: +SKIP

    Multivariate forecast on a real dataset (each column forecast independently):

    >>> from sktime.datasets import load_longley
    >>> _, y_multi = load_longley()  # 5-column economic indicators frame
    >>> y_multi_train = y_multi.iloc[:-3]
    >>> forecaster = T0Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_multi_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["siddharth7113"],
        "maintainers": ["siddharth7113"],
        "python_dependencies": ["tfc-t0"],
        "python_version": ">=3.11",
        # estimator type / capability tags
        # --------------------------------
        "property:randomness": "random",
        "capability:random_state": True,
        "capability:multivariate": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str | None = "theforecastingcompany/t0-alpha",
        device: str | None = None,
        context_length: int | None = None,
        random_state: int | None = None,
        license_accepted: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.random_state = random_state
        self.license_accepted = license_accepted
        self.ignore_deps = ignore_deps

        self.model_ = None

        super().__init__()

        if ignore_deps:
            self.set_tags(python_dependencies=[])

    def __post_init__(self):
        """Post-init constructor logic: resolve device and random state.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        if self.random_state is None:
            self._random_state = np.random.randint(0, 2**31)
        else:
            self._random_state = self.random_state

        if self.device is not None:
            self._device = self.device
        else:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getstate__(self):
        """Return state for pickling, excluding the unpickleable model."""
        state = self.__dict__.copy()
        if hasattr(self, "model_"):
            state["model_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_model_kwargs(self):
        """Collect arguments that uniquely identify a loaded model instance."""
        return {
            "pretrained_model_name_or_path": self.model_path,
            "device": self._device,
        }

    def _get_unique_key(self):
        """Build a cache key from the model-identifying arguments."""
        return str(sorted(self._get_model_kwargs().items()))

    def _load_model(self):
        """Load the (cached) T0 model for the current configuration."""
        return _CachedT0(
            key=self._get_unique_key(),
            t0_kwargs=self._get_model_kwargs(),
        ).load_from_checkpoint()

    def _ensure_model_loaded(self):
        """Reload the model if needed, e.g. after unpickling."""
        if not hasattr(self, "model_") or self.model_ is None:
            self.model_ = self._load_model()

    @classmethod
    def print_license(cls):
        """Print the license and notice shipped with the ``tfc-t0`` package."""
        import importlib

        dist = importlib.metadata.distribution("tfc-t0")
        for fname in ("licenses/LICENSE", "licenses/NOTICE"):
            text = dist.read_text(fname)
            if text:
                print(text)

    def _check_license(self):
        """Guard loading of a gated checkpoint behind explicit license acceptance.

        Raises if a real checkpoint is requested (``model_path`` is not ``None``)
        but the user has not set ``license_accepted=True``. The local random
        model (``model_path=None``) needs no license and is exempt.
        """
        if self.model_path is not None and not self.license_accepted:
            raise ValueError(
                f"Use of the T0 checkpoint '{self.model_path}' is subject to the "
                "license and access terms of its vendor, The Forecasting Company. "
                "The default checkpoint is a gated model on the HuggingFace Hub: "
                "access requires accepting the vendor's terms on the model page "
                "and authenticating with a HuggingFace token. You must read and "
                "accept those terms to use it. "
                "To confirm that you have read and accepted the license terms, set "
                "the `license_accepted` parameter to True. "
                "To print and view the license shipped with the t0 package, call "
                "`T0Forecaster.print_license()`. "
                "Alternatively, set `model_path=None` to build a small "
                "randomly-initialized model locally (untrained; for testing only)."
            )

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._check_license()
        self.model_ = self._load_model()

        context = y
        if self.context_length is not None and context.shape[0] > self.context_length:
            context = context.iloc[-self.context_length :]

        # T0 expects context as (n_series, n_timesteps); each y column is a series
        self.context_ = context.values.T
        self._y_index_names = y.index.names
        return self

    def _predict(self, fh, X):
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
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        out, index, pred_out = self._run_forecast(fh, X, quantiles=[0.5])

        # median point forecast, shape (n_series, horizon)
        point_forecast = np.asarray(out.median)

        pred_df = pd.DataFrame(
            point_forecast.T,
            index=index,
            columns=self._get_varnames(),
        )
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # T0 requires quantile levels strictly in (0, 1), sorted and unique
        quantile_levels = sorted(set(alpha))
        out, index, pred_out = self._run_forecast(fh, X, quantiles=quantile_levels)

        # quantiles array shape (n_series, horizon, n_quantile_levels)
        q_values = np.asarray(out.quantiles)
        var_names = self._get_varnames()

        columns = pd.MultiIndex.from_product([var_names, alpha])
        data = {}
        for s, var in enumerate(var_names):
            for a in alpha:
                level_idx = quantile_levels.index(a)
                data[(var, a)] = q_values[s, :, level_idx]

        pred_df = pd.DataFrame(data, index=index, columns=columns)
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _run_forecast(self, fh, X, quantiles):
        """Run the T0 model for ``fh`` and return the forecast plus index helpers.

        Shared by ``_predict`` and ``_predict_quantiles``: loads the model, builds
        the context and (optional) covariate tensors, and calls ``model.predict``.

        Returns
        -------
        out : t0.model.model.Forecast
            The raw T0 forecast (``out.median``, ``out.quantiles``).
        index : pd.Index
            The absolute time index spanning steps 1..horizon from the cutoff.
        pred_out : pd.Index
            The subset of ``index`` actually requested by ``fh``.
        """
        import torch

        self._ensure_model_loaded()

        horizon = int(max(fh.to_relative(self.cutoff)))

        context = torch.as_tensor(self.context_, dtype=torch.float32)
        n_series, context_len = context.shape

        predict_kwargs = {}
        future_covariates = self._build_future_covariates(
            X, n_series, context_len, horizon
        )
        if future_covariates is not None:
            predict_kwargs["future_covariates"] = future_covariates

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._random_state)
            out = self.model_.predict(
                context, horizon=horizon, quantiles=quantiles, **predict_kwargs
            )

        index = (
            ForecastingHorizon(range(1, horizon + 1)).to_absolute(self._cutoff)._values
        )
        pred_out = fh.get_expected_pred_idx(self.context_, cutoff=self.cutoff)
        return out, index, pred_out

    def _build_future_covariates(self, X, n_series, context_len, horizon):
        """Assemble the ``future_covariates`` tensor for the T0 predict call.

        T0 requires covariates spanning the context window and the horizon, i.e.
        shape ``(n_series, n_covariates, context_len + horizon)``. The context part
        is taken from the exogenous data seen in ``fit`` (``self._X``), the horizon
        part from the exogenous data ``X`` passed to ``predict``.

        Returns ``None`` if no exogenous data is available.
        """
        if X is None:
            return None

        import torch

        if self._X is None:
            raise ValueError(
                "X was not provided in fit but is provided in predict. "
                "To use future covariates, provide past covariate values "
                "in fit as well."
            )

        past = self._X.values[-context_len:]  # (context_len, n_covariates)
        future = X.values[:horizon]  # (horizon, n_covariates)
        covariates = np.concatenate([past, future], axis=0)  # (ctx+horizon, n_cov)

        # to (n_covariates, ctx+horizon), then broadcast across the series batch
        covariates = covariates.T
        covariates = np.broadcast_to(
            covariates[None, :, :],
            (n_series, covariates.shape[0], context_len + horizon),
        )
        return torch.as_tensor(covariates, dtype=torch.float32)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # model_path=None builds a small random-weight model locally, so the
        # contract suite runs without downloading the gated t0-alpha checkpoint
        # (forked-PR CI cannot authenticate to the HuggingFace Hub)
        return [
            {"model_path": None},
            {"model_path": None, "random_state": 42},
        ]


@_multiton
class _CachedT0:
    """Cached T0 model to ensure only one instance exists in memory.

    T0 is a zero-shot model and immutable, so sharing the same instance
    across multiple uses has no side effects.
    """

    def __init__(self, key, t0_kwargs):
        self.key = key
        self.t0_kwargs = t0_kwargs
        self.model = None

    def load_from_checkpoint(self):
        """Load the T0 model (pretrained checkpoint, or a local random model)."""
        if self.model is not None:
            return self.model

        import torch
        from t0 import T0Forecaster as _T0Model

        model_path = self.t0_kwargs["pretrained_model_name_or_path"]
        device = self.t0_kwargs.get("device", "cpu")

        if model_path is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = _T0Model(**_RANDOM_MODEL_CONFIG)
            self.model = model.eval().to(device)
        else:
            # load weights straight onto the target device (faster than
            # load-then-.to); tfc-t0 uses HF Hub's PyTorchModelHubMixin, whose
            # kwarg is ``map_location``
            self.model = _T0Model.from_pretrained(
                model_path, map_location=device
            ).eval()
        return self.model
