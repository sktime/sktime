# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Sundial forecaster for ``sktime``."""

__author__ = ["WenWeiTHU", "geetu040"]
# WenWeiTHU for thuml/sundial-base-128m

__all__ = ["SundialForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.singleton import _multiton


def _empirical_quantiles_1d(samples, alpha):
    """Empirical quantiles matching ``skpro`` ``Empirical._ppf_np`` uniform weights."""
    spl = np.sort(samples)
    weights = np.ones(len(spl), dtype=float)
    weights = weights / weights.sum()
    cum_weights = np.cumsum(weights)
    return np.array([spl[np.searchsorted(cum_weights, a)] for a in alpha])


class SundialForecaster(BaseForecaster):
    """Sundial forecaster via Hugging Face ``transformers``.

    This forecaster wraps Sundial [1]_, [2]_, [3]_ and exposes forecasting
    through the ``sktime`` forecasting interface. Calling :meth:`fit` loads the
    model and stores the observed series as forecasting context. Calling
    :meth:`pretrain` fine-tunes the model on panel or hierarchical data through
    the Sundial forward loss.

    Sundial generates one or more sample paths. Point forecasts are computed as
    the empirical mean over generated samples. Quantile forecasts are computed
    directly from generated samples without requiring ``skpro``. When ``skpro``
    is available, ``predict_proba`` returns an ``Empirical`` distribution over
    the same sample paths; quantiles from ``predict_proba().quantile(...)`` and
    ``predict_quantiles`` describe the same stepwise empirical distribution.

    Parameters
    ----------
    model_path : str, default="thuml/sundial-base-128m"
        Hugging Face repository identifier or local path to a Sundial
        checkpoint. If ``None``, a model is created from ``config`` with random
        weights and should be pretrained before it is used for meaningful
        forecasting.
    config : SundialConfig or dict, optional (default=None)
        Model configuration used to initialize or override the Sundial model. If
        provided as a ``dict``, it is converted with
        ``SundialConfig.from_dict``. If ``model_path=None``, the model is
        initialized from this config with random weights. If ``model_path`` is a
        checkpoint and ``config`` changes parameter shapes, incompatible
        checkpoint weights are initialized from scratch; pretraining or
        fine-tuning is recommended before forecasting.
    device : str, int, or torch.device, default="cpu"
        Device on which to place the model, for example ``"cpu"``,
        ``"cuda"``, or ``"cuda:0"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model.generate(...)`` during
        prediction. Sundial-specific options include arguments such as
        ``num_samples`` and ``revin``; standard generation options supported by
        ``transformers.GenerationMixin.generate`` may also be passed. See the
        Sundial model card [3]_ and Transformers generation docs [4]_ for
        details.
    random_state : int, RandomState instance or None, default=None
        Random seed used for Sundial sampling during prediction. If set,
        repeated predictions from the same fitted state are reproducible.
    validation_split : float or None, default=0.2
        Fraction of data reserved for evaluation when :meth:`pretrain` is used.
        If ``None``, no evaluation dataset is created.
    training_args : dict, optional (default=None)
        Keyword arguments used to construct ``transformers.TrainingArguments``
        in :meth:`pretrain` [5]_.
    compute_metrics : callable or dict, optional (default=None)
        Metrics callback(s) passed to ``transformers.Trainer`` [5]_.
    callbacks : list, optional (default=None)
        Trainer callbacks passed to ``transformers.Trainer`` [5]_.

    References
    ----------
    .. [1] Sundial: A Family of Highly Capable Time Series Foundation Models:
       https://arxiv.org/abs/2502.00816
    .. [2] Sundial repository:
       https://github.com/thuml/Sundial
    .. [3] Sundial model card:
       https://huggingface.co/thuml/sundial-base-128m
    .. [4] Transformers `.generate()`:
       https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate
    .. [5] Trainer/TrainingArguments docs:
       https://huggingface.co/docs/transformers/en/main_classes/trainer

    Examples
    --------
    Simple zero-shot forecasting with thuml/sundial-base-128m:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = SundialForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Running with explicit device, dtype, and sampling settings:

    >>> import torch  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     device="cuda",
    ...     dtype=torch.bfloat16,
    ...     forward_kwargs={"num_samples": 20},
    ...     random_state=42,
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Passing Sundial and Transformers generation options through
    ``forward_kwargs``:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 20, "revin": False},
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Quantile prediction from generated samples:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 50},
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )

    Global training on panel data before forecasting a single series:

    The example below changes ``output_token_lens``. This can initialize the
    affected prediction-head weights from scratch, so :meth:`pretrain` is run
    before forecasting.

    >>> import torch  # doctest: +SKIP
    >>> from sktime.datasets import load_airline, load_tecator
    >>> from sktime.forecasting.sundial import SundialForecaster  # doctest: +SKIP
    >>> device = "cuda" if torch.cuda.is_available() else None  # doctest: +SKIP
    >>> y_panel = load_tecator(  # doctest: +SKIP
    ...     return_type="pd-multiindex",
    ...     return_X_y=False,
    ... )
    >>> y_panel = y_panel.drop(["class_val"], axis=1)  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     training_args={
    ...         "output_dir": "sundial-output",
    ...         "do_train": True,
    ...         "do_eval": True,
    ...         "evaluation_strategy": "epoch",
    ...         "num_train_epochs": 10,
    ...         "per_device_train_batch_size": 8,
    ...         "per_device_eval_batch_size": 8,
    ...         "learning_rate": 5e-5,
    ...     },
    ...     config={"output_token_lens": [8]},
    ...     device=device,
    ...     dtype=torch.bfloat16,
    ...     validation_split=0.3,
    ... )
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pretrain": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "authors": ["WenWeiTHU", "geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]~=4.40.0"],
        "tests:vm": True,
        "tests:libs": ["sktime.libs.sundial"],
    }

    def __init__(
        self,
        model_path="thuml/sundial-base-128m",
        config=None,
        device="cpu",
        dtype=None,
        forward_kwargs=None,
        random_state=None,
        validation_split=0.2,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.dtype = dtype
        self.forward_kwargs = forward_kwargs
        self.random_state = random_state
        self.validation_split = validation_split
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks

        super().__init__()

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain forecaster on panel/global data (first batch).

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets pretrained model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex (guaranteed Panel or Hierarchical)
            Panel or hierarchical time series data to pretrain on.
            The last index level is time, all other levels identify instances.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series.
        fh : ForecastingHorizon or None, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        self.model_ = self._load_model()

        input_token_len = self.model_.config.input_token_len
        output_token_len = self.model_.config.output_token_lens[-1]

        horizon_length = _get_horizon_length(fh, output_token_len)
        if horizon_length > output_token_len:
            raise ValueError(
                "Requested pretraining horizon exceeds Sundial model capacity: "
                f"max requested step={horizon_length}, "
                f"max output_token_lens={output_token_len}. "
                "Use a shorter fh or a model/config with a larger "
                "output_token_lens value."
            )

        from transformers import Trainer, TrainingArguments

        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
        else:
            y_train = y
            y_eval = None

        min_series_length = input_token_len + horizon_length
        train_series_list = [
            series
            for series in _prepare_series_list(y_train)
            if len(series) >= min_series_length
        ]
        if self.validation_split is not None and len(train_series_list) == 0:
            full_series_list = [
                series
                for series in _prepare_series_list(y)
                if len(series) >= min_series_length
            ]
            if len(full_series_list) > 0:
                train_series_list = full_series_list
                y_eval = None
                warn(
                    "Skipping Sundial evaluation dataset creation: validation "
                    "split leaves no numeric series long enough to create a "
                    "training target. Training continues on the full input. "
                    "You can also decrease output_token_lens through config "
                    "to reduce the default pretraining horizon.",
                    stacklevel=2,
                )

        train = SundialPyTorchDataset(
            series_list=train_series_list,
            min_length=min_series_length,
        )

        eval = None
        if y_eval is not None:
            eval_series_list = [
                series
                for series in _prepare_series_list(y_eval)
                if len(series) >= min_series_length
            ]

        if y_eval is not None and len(eval_series_list) > 0:
            eval = SundialPyTorchDataset(
                series_list=eval_series_list,
                min_length=min_series_length,
            )
        elif y_eval is not None:
            warn(
                "Skipping Sundial evaluation dataset creation: validation split "
                "does not contain any numeric series long enough to create a "
                "training target. Training continues without evaluation. You "
                "can also decrease output_token_lens through config to reduce "
                "the default pretraining horizon.",
                stacklevel=2,
            )

        data_collator = SundialDataCollator(
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            horizon_length=horizon_length,
            dtype=self.model_.dtype,
        )

        training_args = (
            deepcopy(self.training_args) if self.training_args is not None else {}
        )
        training_args = TrainingArguments(**training_args)

        self.model_.train()
        trainer = Trainer(
            model=self.model_,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )
        trainer.train()

        self.model_ = trainer.model

    def _fit(self, y, X=None, fh=None):
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
        self.model_ = self._load_model()
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
        self.model_.eval()
        samples, fh, preds_idx = self._generate_samples(fh)

        preds = samples.mean(axis=1)
        preds = preds[:, preds_idx].T
        preds = pd.DataFrame(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=self.context_.columns,
        )

        return preds

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
        samples, fh, preds_idx = self._generate_samples(fh)

        alpha = [round(i, 3) for i in alpha]
        preds = samples[:, :, preds_idx]
        n_series, _, n_horizon = preds.shape
        quantiles = np.empty((n_series, n_horizon, len(alpha)))
        for i in range(n_series):
            for j in range(n_horizon):
                quantiles[i, j] = _empirical_quantiles_1d(preds[i, :, j], alpha)
        quantiles = quantiles.transpose(1, 0, 2).reshape(n_horizon, -1)
        quantiles = quantiles.astype(float)

        columns = pd.MultiIndex.from_product([self.context_.columns, alpha])
        pred_quantiles = pd.DataFrame(
            data=quantiles,
            index=fh.to_absolute(self._cutoff)._values,
            columns=columns,
        )

        return pred_quantiles

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts from generated samples.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : skpro.distributions.empirical.Empirical
            predictive distribution from generated sample paths
        """
        from skpro.distributions.empirical import Empirical

        samples, fh, preds_idx = self._generate_samples(fh)

        preds = samples[:, :, preds_idx]
        pred_index = fh.to_absolute_index(self.cutoff)
        n_series, n_samples, _ = preds.shape
        sample_index = pd.MultiIndex.from_product(
            [range(n_samples), pred_index],
            names=["sample", *pred_index.names],
        )
        samples_df = pd.DataFrame(
            preds.transpose(1, 2, 0).reshape(-1, n_series),
            index=sample_index,
            columns=self.context_.columns,
        )

        return Empirical(
            samples_df,
            time_indep=marginal,
            index=pred_index,
            columns=self.context_.columns,
        )

    def _generate_samples(self, fh):
        """Generate raw Sundial sample paths for prediction.

        This helper converts the fitted forecasting context to Sundial's
        expected tensor layout, validates that the requested horizon is within
        the configured ``output_token_lens`` capacity, and calls
        ``model.generate`` with ``forward_kwargs``.

        Parameters
        ----------
        fh : ForecastingHorizon or None
            Forecasting horizon requested by ``predict`` or
            ``predict_quantiles``. If ``None``, ``self.fh`` is used.

        Returns
        -------
        samples : np.ndarray
            Generated sample paths with shape
            ``(n_series, n_samples, horizon_length)``.
        fh : ForecastingHorizon
            Relative forecasting horizon used for generation.
        preds_idx : np.ndarray
            Zero-based positions in ``samples`` corresponding to the requested
            forecasting horizon.
        """
        import torch
        from sklearn.utils import check_random_state

        self.model_ = self._load_model()
        self.model_.eval()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon_length = np.max(preds_idx) + 1
        output_token_lens = getattr(self.model_.config, "output_token_lens", [])
        if output_token_lens and max(output_token_lens) < horizon_length:
            raise ValueError(
                "Requested forecasting horizon exceeds Sundial model capacity: "
                f"max requested step={horizon_length}, "
                f"max output_token_lens={max(output_token_lens)}. "
                "Use a shorter horizon or a model/config with a larger "
                "output_token_lens value."
            )

        past_values = self.context_.to_numpy().T
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        if self.random_state is None:
            output = self.model_.generate(
                past_values,
                max_new_tokens=horizon_length,
                **forward_kwargs,
            )
        else:
            seed = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
            devices = []
            if past_values.device.type == "cuda":
                devices = [past_values.device]

            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                output = self.model_.generate(
                    past_values,
                    max_new_tokens=horizon_length,
                    **forward_kwargs,
                )

        samples = output.detach().float().cpu().numpy()

        return samples, fh, preds_idx

    def _load_model(self):
        """Load or retrieve a cached Sundial model instance."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedSundial(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device": self.device,
            "dtype": self.dtype,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skbase.utils.dependencies import _check_soft_dependencies

        common_params = {
            "model_path": None,
            "config": {
                "input_token_len": 2,
                "hidden_size": 4,
                "intermediate_size": 8,
                "output_token_lens": [8],
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "max_position_embeddings": 64,
                "flow_loss_depth": 1,
                "num_sampling_steps": 1,
                "diffusion_batch_mul": 1,
                "use_cache": False,
            },
            "random_state": 42,
            "validation_split": 0.1,
            "training_args": {
                "output_dir": "test_output",
                "max_steps": 1,
            },
        }

        test_params_1 = {
            **common_params,
            "device": "cpu",
            "forward_kwargs": {"num_samples": 2, "revin": True},
        }

        test_params_2 = {
            **common_params,
            "forward_kwargs": {"num_samples": 3, "revin": False},
        }

        if _check_soft_dependencies("torch", severity="none"):
            import torch

            test_params_2.update(
                {
                    "dtype": torch.bfloat16,
                    "device": "cuda" if torch.cuda.is_available() else None,
                }
            )

        params = [test_params_1, test_params_2]

        return params


@_multiton
class _CachedSundial:
    """Multiton-backed cache wrapper for a loaded Sundial model."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device,
        dtype,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance."""
        if self.model_ is not None:
            return self.model_

        if self.model_path is None:
            warn(
                "Initializing Sundial from config creates random weights. Sundial "
                "pretraining is required before these weights are suitable for "
                "meaningful forecasting.",
                UserWarning,
                stacklevel=2,
            )

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        self.model_ = self.model_.to(self.device, dtype=self.dtype)

        return self.model_

    def _load_from_path(self):
        """Load Sundial model weights from ``self.model_path``."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        config = deepcopy(self.config)
        if isinstance(config, dict):
            config = SundialConfig.from_dict(config)

        return SundialForPrediction.from_pretrained(
            self.model_path, config=config, ignore_mismatched_sizes=True
        )

    def _load_randomly(self):
        """Initialize a Sundial model randomly from config."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        config = deepcopy(self.config)
        if not config:
            config = SundialConfig()
        if isinstance(config, dict):
            config = SundialConfig.from_dict(config)

        return SundialForPrediction(config)


def _get_horizon_length(fh, output_token_len):
    """Return pretraining horizon length from fh or model output capacity."""
    if fh is None:
        return output_token_len
    fh = fh.to_relative(cutoff=0)
    return int(np.max(fh._values.values))


def _prepare_series_list(data):
    """Convert panel/hierarchical DataFrame into 1D training series."""
    if data.index.nlevels == 1:
        groups = [(None, data)]
    else:
        instance_levels = list(range(data.index.nlevels - 1))
        groupby_level = (
            instance_levels[0] if len(instance_levels) == 1 else instance_levels
        )
        groups = data.groupby(level=groupby_level)

    series_list = []
    for _, group in groups:
        for col in group.columns:
            values = group[col].to_numpy(dtype=np.float32)
            values = values[np.isfinite(values)]
            series_list.append(values)

    return series_list


class SundialPyTorchDataset:
    """Dataset for Sundial pretraining with ``Trainer``."""

    def __init__(self, series_list, min_length=1):
        self.min_length = min_length
        self.series_list = [
            series for series in series_list if len(series) >= self.min_length
        ]

        if not self.series_list:
            raise ValueError(
                "No training series were available for Sundial pretraining. "
                "Provide at least one numeric series with length greater than "
                f"or equal to {self.min_length}. You can also decrease "
                "output_token_lens through config to reduce the default "
                "pretraining horizon."
            )

    def __len__(self):
        """Return number of training series."""
        return len(self.series_list)

    def __getitem__(self, i):
        """Return one unpadded training series."""
        import torch

        return torch.tensor(self.series_list[i], dtype=torch.float32)


class SundialDataCollator:
    """Pad Sundial training series to a common batch shape."""

    def __init__(self, input_token_len, output_token_len, horizon_length, dtype=None):
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.horizon_length = horizon_length
        self.dtype = dtype

    def __call__(self, features):
        """Collate variable-length series into Sundial training tensors."""
        import torch

        lengths = [len(feature) for feature in features]
        max_length = max(lengths)
        min_length = min(lengths)

        min_required_length = self.input_token_len + self.horizon_length
        if min_length < min_required_length:
            raise ValueError(
                "Sundial pretraining requires every series in a batch to be at "
                "least as long as input_token_len + horizon_length. Found "
                f"length {min_length}, but input_token_len={self.input_token_len} "
                f"and horizon_length={self.horizon_length}. You can also "
                "decrease output_token_lens through config to reduce the "
                "default pretraining horizon."
            )

        input_length = max_length - self.horizon_length
        n_input_tokens = (
            input_length + self.input_token_len - 1
        ) // self.input_token_len
        model_left_pad = n_input_tokens * self.input_token_len - input_length
        label_right_pad = model_left_pad + self.output_token_len - self.horizon_length

        input_ids = []
        labels = []
        loss_masks = []
        for feature, length in zip(features, lengths):
            left_pad = max_length - length
            feature = torch.nn.functional.pad(feature, (left_pad, 0), value=0.0)

            input_ids.append(feature[: -self.horizon_length])
            labels.append(
                torch.nn.functional.pad(
                    feature[self.input_token_len :],
                    (0, label_right_pad),
                    value=0.0,
                )
            )

            token_starts = np.arange(n_input_tokens) * self.input_token_len
            token_starts = token_starts - model_left_pad - left_pad
            valid_targets = token_starts + self.horizon_length <= length
            valid_targets = valid_targets & (token_starts >= 0)
            loss_masks.append(torch.tensor(valid_targets, dtype=torch.float32))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        if self.dtype is not None:
            input_ids = input_ids.to(dtype=self.dtype)
            labels = labels.to(dtype=self.dtype)

        mask_y = torch.zeros(len(features), self.output_token_len, dtype=torch.float32)
        mask_y[:, : self.horizon_length] = 1.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": torch.stack(loss_masks),
            "mask_y": mask_y,
        }
