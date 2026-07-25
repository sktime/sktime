# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

__author__ = ["ajati", "wgifford", "vijaye12", "geetu040"]
# ajati, wgifford, vijaye12 for ibm-granite code

import warnings

import numpy as np
import pandas as pd
from skbase.utils.stdout_mute import StdoutMute

from sktime.forecasting.base import (
    BaseForecaster,
    ForecastingHorizon,
    _GlobalForecastingDeprecationMixin,
)
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import
from sktime.utils.singleton import _multiton
from sktime.utils.warnings import warn

torch = _safe_import("torch")
Dataset = _safe_import("torch.utils.data.Dataset")

_FREQUENCY_TOKEN_MAP = {
    "oov": 0,
    "min": 1,
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,
    "d": 8,
    "w": 9,
}


class TinyTimeMixerForecaster(_GlobalForecastingDeprecationMixin, BaseForecaster):
    """
    TinyTimeMixer Forecaster for Zero-Shot Forecasting of Multivariate Time Series.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for tutorial by creators.

    TinyTimeMixer (TTM) are compact pre-trained models for Time-Series Forecasting,
    open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces
    the notion of the first-ever "tiny" pre-trained models for Time-Series Forecasting.

    **Fit Strategies: Full, Minimal, and Zero-shot**

    This model supports three fit strategies: *zero-shot* for direct predictions without
    training, *minimal* fine-tuning for lightweight adaptation to new data, and *full*
    fine-tuning for comprehensive model training. The selected strategy is determined
    by the model's fit_strategy parameter

    **Initialization Process**:

    1. **Model Path**: The ``model_path`` parameter points to a local folder or
       Hugging Face repo that contains both *configuration files*
       and *pretrained weights*. Public TinyTimeMixer checkpoints are available
       for the R1 [4]_, R2 [5]_, research R2 [6]_, and R3 [7]_ model families.

    2. **Default Configuration**: The model loads its default configuration from the
       *configuration files*.

    3. **Custom Configuration**: Users can provide a custom configuration via the
       ``config`` parameter during model initialization.

    4. **Configuration Override**: If custom configuration is provided,
       it overrides the default configuration.

    5. **Forecasting Horizon**: If ``revision=None``, the forecasting horizon
       (``fh``) specified during ``fit`` is used to select a compatible
       checkpoint revision. For prediction, the requested horizon must fit
       within the loaded model's ``config.prediction_length``.

    6. **Model Architecture**: The final configuration is used to construct the
       *model architecture*.

    7. **Pretrained Weights**: *pretrained weights* are loaded from the ``model_path``,
       these weights are then aligned and loaded into the *model architecture*.

    8. **Weight Alignment**: However sometimes, *pretrained weights* do not align with
       the *model architecture*, because the config was changed which created a
       *model architecture* of different size than the default one.
       This causes some of the weights in *model architecture* to be reinitialized
       randomly instead of using the pre-trained weights.

    **Training Strategies**:

    - **Zero-shot Forecasting**: When all the *pre-trained weights* are correctly
      aligned with the *model architecture*, fine-tuing part is bypassed and
      the model preforms zero-short forecasting.

    - **Minimal Fine-tuning**: When not all the *pre-trained weights* are correctly
      aligned with the *model architecture*, rather some weights are re-initialized,
      these re-initialized weights are fine-tuned on the provided data.

    - **Full Fine-tuning**:  The model is *fully fine-tuned* on new data, updating *all
      parameters*. This approach offers maximum adaptation to the dataset but requires
      more computational resources.

    **Exogenous Variables Support**

    TTM supports exogenous variables (external factors) that can improve
    forecasting accuracy. The model accepts exogenous variables that are
    known for both the historical period and the future forecasting horizon.

    When using exogenous variables:
    - The X parameter should contain exogenous data covering both
      past and future periods
    - Exogenous variables must have the same index structure as the target series
    - For prediction, exogenous data must extend into the forecasting horizon

    Parameters
    ----------
    model_path : str, default="ibm/TTM"
        Path to the Hugging Face model to use for forecasting.
        This can be either:

        - The name of a Hugging Face repository, for example
          ``"ibm-research/ttm-r3"`` [7]_. Related checkpoints are listed in
          the TinyTimeMixer [8]_, Granite time series [9]_, and IBM Research
          time series [10]_ collections.

        - A local path to a folder containing model files in a format supported
          by transformers. In this case, ensure that the directory contains all
          necessary files (e.g., configuration, tokenizer, and model weights).

        - If this parameter is *None*, fit_strategy should be *full* to allow
          training a randomly initialized model from the provided or default
          config, else ValueError is raised.

    revision : str or None, default="main"
        Revision of the model to use:

        - None: Automatically select a compatible revision based on the
          training context length and forecasting horizon.

        - "main": Load the main branch of the selected checkpoint.

        - A checkpoint branch name such as "52-16-ft-r2.1" can be used to load
          a specific TinyTimeMixer variant.

        This param becomes irrelevant when model_path is None.

    device : str, default="cpu"
        Device for model inference and fine-tuning, for example ``"cpu"``,
        ``"cuda"``, or ``"cuda:0"``.

    freq : str or None, default=None
        Frequency to pass to models that use resolution prefix tuning,
        such as TTM-R2 [5]_ and research R2 [6]_. If ``None``, the frequency
        is inferred from the forecasting horizon or time index where possible,
        and falls back to the
        out-of-vocabulary token.

    verbose : bool, default=False
        If True, show training output from ``transformers.Trainer``.

    validation_split : float, default=0.2
        Fraction of the data to use for validation

    config : dict or None, default={}
        Configuration to use for the model. See the ``transformers``
        documentation for details. The provided configuration must be valid for
        the selected TinyTimeMixer model architecture.

    training_args : dict or None, default={}
        Training arguments to use for the model. See ``transformers.TrainingArguments``
        for details.
        Note that the ``output_dir`` argument is required.

    compute_metrics : list, default=[]
        List of metrics to compute during training. See ``transformers.Trainer``
        for details.

    callbacks : list, default=None
        List of callbacks to use during training. See ``transformers.Trainer``

    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.

    use_source_package : bool, default=False
        If True, the model and configuration will be loaded directly from the source
        package ``tsfm_public.models.tinytimemixer``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime because of model's unavailability
        on pypi.
        To install the source package, follow the instructions here [1]_.

    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:
        - "zero-shot": Uses pre-trained model as it is. If model path is *None*
          with this strategy, ValueError is raised.
        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
          If model path is *None* with this strategy, ValueError is raised.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time. Allows
          model path to be *None*.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/
    .. [2] Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S.,
           Nguyen, N.H., Gifford, W.M., Reddy, C. and Kalagnanam, J., 2024.
           Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced
           Zero/Few-Shot Forecasting of Multivariate Time Series. CoRR.
    .. [3] https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks
    .. [4] https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1
    .. [5] https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
    .. [6] https://huggingface.co/ibm-research/ttm-research-r2
    .. [7] https://huggingface.co/ibm-research/ttm-r3
    .. [8] https://huggingface.co/collections/geetu040/tinytimemixer
    .. [9] https://huggingface.co/collections/ibm-granite/granite-time-series
    .. [10] https://huggingface.co/collections/ibm-research/time-series-models

    Examples
    --------
    Zero-shot forecasting with a pretrained TinyTimeMixer R3 checkpoint [7]_.
    Other supported public checkpoints include R1 [4]_, R2 [5]_, and research
    R2 [6]_ variants:

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path="ibm-research/ttm-r3",
    ...     # Other supported public checkpoints include:
    ...     # model_path="ibm-granite/granite-timeseries-ttm-r1",
    ...     # model_path="ibm-granite/granite-timeseries-ttm-r2",
    ...     # model_path="ibm-granite/granite-timeseries-ttm-r2",
    ...     # revision="52-16-ft-r2.1",
    ...     # model_path="ibm-research/ttm-research-r2",
    ... )
    >>> forecaster.fit(y)
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])

    Automatically select the best compatible model revision from the context
    length and prediction length:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path="ibm-research/ttm-r3",
    ...     revision=None,
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict()

    Forecasting with exogenous variables known during training and prediction:

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_longley
    >>> from sktime.split import temporal_train_test_split
    >>> y, X = load_longley()
    >>> y_train, _, X_train, X_future = temporal_train_test_split(y, X, test_size=2)
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path="ibm-research/ttm-r3",
    ... )
    >>> forecaster.fit(y_train, X=X_train, fh=[1, 2])
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict(X=X_future)

    Minimal fine-tuning updates only parameters that are not loaded from the
    checkpoint, for example when the supplied configuration changes the model
    shape:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path="ibm-research/ttm-r3",
    ...     fit_strategy="minimal",
    ...     config={
    ...         "context_length": 24,
    ...         "trend_patch_length": 6,
    ...         "trend_patch_stride": 6,
    ...         "prediction_length": 12,
    ...     },
    ...     training_args={
    ...         "max_steps": 10,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 4,
    ...         "report_to": "none",
    ...     },
    ... )
    >>> forecaster.fit(y)
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])

    Initialize a random model when ``model_path`` is ``None``
    and preform full fine-tuning:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path=None,
    ...     fit_strategy="full",
    ...     training_args={
    ...         "max_steps": 10,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 4,
    ...         "report_to": "none",
    ...     },
    ... )
    >>> forecaster.fit(y)
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])

    Pretrain on panel data before fitting to the target forecasting series:

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_airline, load_tecator
    >>> y = load_airline()
    >>> y_panel = load_tecator(
    ...     return_type="pd-multiindex",
    ...     return_X_y=False,
    ... )
    >>> y_panel.drop(["class_val"], axis=1, inplace=True)
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path="ibm-research/ttm-r3",
    ...     fit_strategy="full",
    ...     training_args={
    ...         "max_steps": 10,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 4,
    ...         "report_to": "none",
    ...     },
    ... )
    >>> forecaster.pretrain(y_panel)
    TinyTimeMixerForecaster(...)
    >>> forecaster.fit(y)
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ajati", "wgifford", "vijaye12", "geetu040"],
        # ajati, wgifford, vijaye12 for ibm-granite code
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers", "torch", "accelerate>=0.26.0"],
        # estimator type
        # --------------
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "capability:unequal_length": True,
        "property:randomness": "stochastic",
        "capability:random_state": False,
        "capability:pretrain": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.granite_ttm"],
    }

    def __init__(
        self,
        model_path="ibm/TTM",
        revision="main",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
        use_source_package=False,
        fit_strategy="minimal",
        device="cpu",
        freq=None,
        verbose=False,
    ):
        super().__init__()
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.freq = freq
        self._freq = None
        self._freq_token = None
        self.verbose = verbose
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.validation_split = validation_split
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.broadcasting = broadcasting
        self.use_source_package = use_source_package
        self.fit_strategy = fit_strategy

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

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
        return self._fit_or_pretrain(y=y, X=X, fh=fh)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        return self._fit_or_pretrain(y=y, X=X, fh=fh)

    def _fit_or_pretrain(self, y, X=None, fh=None):
        """Load and optionally train the TTM model for fit or pretrain.

        This contains the common logic used by ``_fit`` and ``_pretrain``.
        It resolves the frequency token and model revision, loads or initializes
        the cached TinyTimeMixer model, prepares train/evaluation datasets, and
        runs ``transformers.Trainer`` when the selected ``fit_strategy`` leaves
        trainable parameters.

        The fitted/pretrained model state produced here is consumed by
        ``_predict``. In particular, the method sets ``self.model_``,
        ``self._freq_token``, and ``self._revision`` so prediction can construct
        padded/truncated context windows, optional exogenous future values, and
        frequency-token inputs for the loaded model.

        Parameters
        ----------
        y : sktime time series object
            Target series used for fitting or pretraining. For ``_pretrain``,
            this may be panel or hierarchical data with the last index level
            representing time and preceding levels identifying instances.
        X : sktime time series object, optional (default=None)
            Exogenous time series aligned with ``y``. If provided, it is split
            consistently with ``y`` for validation and passed to the PyTorch
            dataset used by the trainer.
        fh : ForecastingHorizon or None, optional (default=None)
            Forecasting horizon used to resolve automatic model revisions and
            later consumed by prediction. Required when ``revision=None``.
        """
        self._freq, self._freq_token = self._resolve_freq(y=y, fh=fh)
        self._revision = self._resolve_revision(y=y, fh=fh)

        self.model_ = _CachedTinyTimeMixer(
            key=self._get_unique_key(),
            model_path=self.model_path,
            revision=self._revision,
            device=self.device,
            user_config=self._config,
            use_source_package=self.use_source_package,
            fit_strategy=self.fit_strategy,
        ).load()

        if not self.model_.config.resolution_prefix_tuning:
            self._freq_token = None

        if not any(param.requires_grad for param in self.model_.parameters()):
            return

        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
            # Handle exogenous variables split if provided
            X_train, X_eval = None, None
            if X is not None:
                X_train, X_eval = temporal_train_test_split(
                    X, test_size=self.validation_split
                )
        else:
            y_train = y
            y_eval = None
            X_train = X
            X_eval = None

        train = PyTorchDataset(
            y=y_train,
            context_length=self.model_.config.context_length,
            prediction_length=self.model_.config.prediction_length,
            X=X_train,
            frequency_token=self._freq_token,
        )

        eval = None
        if self.validation_split is not None:
            eval = PyTorchDataset(
                y=y_eval,
                context_length=self.model_.config.context_length,
                prediction_length=self.model_.config.prediction_length,
                X=X_eval,
                frequency_token=self._freq_token,
            )

        from transformers import Trainer, TrainingArguments

        # Get Training Configuration
        training_args = TrainingArguments(**self._training_args)

        # Get the Trainer
        trainer = Trainer(
            model=self.model_,
            args=training_args,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        if self.verbose:
            trainer.train()
        else:
            with StdoutMute() as _:
                trainer.train()

        # Get the model
        self.model_ = trainer.model

    def _resolve_freq(self, y, fh=None):
        freq = self.freq or getattr(fh, "freq", None) or _infer_index_frequency(y.index)
        freq_token = _map_frequency_token(freq)
        return freq, freq_token

    def _resolve_revision(self, y, fh=None):
        if self.revision is not None or self.model_path is None:
            return self.revision

        if fh is None:
            raise ValueError(
                "revision=None requires `fh` to be passed to `fit`, so the "
                "TinyTimeMixer revision can be selected from the requested "
                "forecasting horizon. Pass `fh` in `fit`, or set `revision` "
                "explicitly."
            )

        context_len = _get_context_length(y)
        horizon_len = fh.to_numpy().max()

        return _get_auto_revision(
            model_path=self.model_path,
            context_len=context_len,
            horizon_len=horizon_len,
        )

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "revision": self._revision,
            "device": self.device,
            "config": self._config,
            "fit_strategy": self.fit_strategy,
            "use_source_package": self.use_source_package,
        }
        return str(sorted(key.items()))

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
            The forecasting horizon with the steps ahead to predict.
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
        import torch

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        max_fh = fh.to_numpy().max()
        prediction_length = self.model_.config.prediction_length
        if prediction_length < max_fh:
            raise ValueError(
                "The requested forecasting horizon is longer than the loaded "
                "TinyTimeMixer model can predict. The maximum requested "
                f"relative horizon is {max_fh}, but the model "
                f"prediction_length is {prediction_length}. Fit the forecaster "
                "with a longer fh, choose a model revision with a longer "
                "prediction_length, or provide a compatible config."
            )

        _y = self._y

        hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps, n_cols)

        # truncate or pad to match sequence length
        past_values, observed_mask = _pad_truncate(
            hist, self.model_.config.context_length
        )

        past_values = (
            torch.tensor(past_values).to(self.model_.dtype).to(self.model_.device)
        )
        observed_mask = (
            torch.tensor(observed_mask).to(self.model_.dtype).to(self.model_.device)
        )

        # Handle exogenous variables if provided
        future_values = None
        if X is not None:
            # Process exogenous variables for prediction
            exog_data = np.expand_dims(X.values, axis=0)

            # Extract future exogenous values for the prediction horizon
            # X should contain future values that cover the prediction horizon
            prediction_length = self.model_.config.prediction_length
            context_length = self.model_.config.context_length

            # Get the last context_length + prediction_length values from exog_data
            if exog_data.shape[1] >= context_length + prediction_length:
                # Take the last prediction_length values as future exogenous
                future_exog = exog_data[:, -prediction_length:, :]
                future_values = (
                    torch.tensor(future_exog)
                    .to(self.model_.dtype)
                    .to(self.model_.device)
                )

        freq_token = None
        if self._freq_token is not None:
            freq_token = torch.tensor(
                [self._freq_token],
                dtype=torch.int,
                device=self.model_.device,
            )

        self.model_.eval()
        outputs = self.model_(
            past_values=past_values,
            past_observed_mask=observed_mask,
            future_values=future_values,
            freq_token=freq_token,
        )
        pred = outputs.prediction_outputs.detach().cpu().numpy()

        # converting pred datatype

        if isinstance(_y.index, pd.MultiIndex):
            ins = np.array(
                list(np.unique(_y.index.droplevel(-1)).repeat(pred.shape[1]))
            )
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y.index.names,
            )
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )

        pred = pd.DataFrame(
            # batch_size * num_timestams, n_cols
            pred.reshape(-1, pred.shape[-1]),
            index=index,
            columns=_y.columns,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]
        pred.index.names = _y.index.names

        return pred

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
        common_params = {
            "config": {
                "context_length": 20,
                "prediction_length": 15,
                "num_patches": 5,
                "patch_length": 4,
                "patch_stride": 4,
                "d_model": 4,
                "num_layers": 1,
                "decoder_d_model": 4,
                "decoder_num_layers": 1,
                "expansion_factor": 1,
                "dropout": 0.0,
                "head_dropout": 0.0,
                "use_decoder": False,
                "loss": "mse",
            },
            "training_args": {
                "max_steps": 1,
                "output_dir": "test_output",
                "per_device_train_batch_size": 4,
                "report_to": "none",
            },
        }
        test_params = [
            {
                "model_path": "ibm-granite/granite-timeseries-ttm-r1",
                "revision": "main",
                "validation_split": 0.4,
                "fit_strategy": "minimal",
                **common_params,
            },
            {
                "model_path": None,
                "validation_split": 0.1,
                "fit_strategy": "full",
                **common_params,
            },
        ]
        return test_params


def _pad_truncate(data, seq_len, pad_value=0):
    """
    Pad or truncate a numpy array.

    Parameters
    ----------
    - data: numpy array of shape (batch_size, original_seq_len, n_dims)
    - seq_len: sequence length to pad or truncate to
    - pad_value: value to use for padding

    Returns
    -------
    - padded_data: array padded or truncated to (batch_size, seq_len, n_dims)
    - mask: mask indicating padded elements (1 for existing; 0 for missing)
    """
    batch_size, original_seq_len, n_dims = data.shape

    # Truncate or pad each sequence in data
    if original_seq_len > seq_len:
        truncated_data = data[:, -seq_len:, :]
        mask = np.ones_like(truncated_data)
    else:
        truncated_data = np.pad(
            data,
            ((0, 0), (seq_len - original_seq_len, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        mask = np.zeros_like(truncated_data)
        mask[:, -original_seq_len:, :] = 1

    return truncated_data, mask


def _infer_index_frequency(index):
    """Infer a pandas-style frequency string from an sktime index."""
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(-1)

    freq = getattr(index, "freqstr", None)
    if freq is not None:
        return freq

    freq = getattr(getattr(index, "freq", None), "freqstr", None)
    if freq is not None:
        return freq

    inferred_freq = getattr(index, "inferred_freq", None)
    if inferred_freq is not None:
        return inferred_freq

    try:
        return pd.infer_freq(index)
    except (TypeError, ValueError):
        return None


def _map_frequency_token(freq):
    """Map a frequency value to the TTM frequency token vocabulary."""
    if freq is None:
        return _FREQUENCY_TOKEN_MAP["oov"]

    if isinstance(freq, (int, np.integer)):
        return int(freq)

    freq = str(freq)
    if freq in _FREQUENCY_TOKEN_MAP:
        return _FREQUENCY_TOKEN_MAP[freq]

    if freq.lower() in _FREQUENCY_TOKEN_MAP:
        return _FREQUENCY_TOKEN_MAP[freq.lower()]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            offset_freq = pd.tseries.frequencies.to_offset(freq).freqstr
    except (TypeError, ValueError):
        offset_freq = None

    if offset_freq in _FREQUENCY_TOKEN_MAP:
        return _FREQUENCY_TOKEN_MAP[offset_freq]
    if offset_freq is not None and offset_freq.lower() in _FREQUENCY_TOKEN_MAP:
        return _FREQUENCY_TOKEN_MAP[offset_freq.lower()]

    return _FREQUENCY_TOKEN_MAP["oov"]


def _get_context_length(y):
    """Return the per-series context length for univariate or panel data."""
    if isinstance(y.index, pd.MultiIndex):
        instance_levels = list(range(y.index.nlevels - 1))
        if instance_levels:
            return int(y.groupby(level=instance_levels, sort=False).size().max())

    return len(y)


def _get_auto_revision(model_path, context_len, horizon_len):
    """Return an auto-selected TTM revision.

    Selects the available Hugging Face branch with the smallest absolute
    distance from the requested context and horizon lengths.
    """
    _check_soft_dependencies(
        "huggingface-hub",
        severity="error",
        msg=(
            "revision=None requires the optional dependency "
            "`huggingface-hub` to inspect available TinyTimeMixer revisions. "
            "Please install `huggingface-hub`, or set `revision` explicitly."
        ),
    )
    from huggingface_hub import HfApi

    api = HfApi()
    refs = api.list_repo_refs(model_path, repo_type="model")
    available_revisions = [i.name for i in refs.branches]

    valid_revisions = []
    for revision in available_revisions:
        parts = revision.split("-")
        if len(parts) < 2:
            continue
        try:
            revision_context_len = int(parts[0])
            revision_horizon_len = int(parts[1])
        except ValueError:
            continue

        distance = int(
            abs(revision_context_len - context_len)
            + abs(revision_horizon_len - horizon_len)
        )

        valid_revisions.append(
            {
                "name": revision,
                "context_len": revision_context_len,
                "horizon_len": revision_horizon_len,
                "distance": distance,
            }
        )

    if not valid_revisions:
        raise ValueError(
            "Could not find valid TinyTimeMixer revisions in "
            f"model_path={model_path!r}."
        )

    valid_revisions = sorted(
        valid_revisions,
        key=lambda revision: revision["distance"],
    )
    selected_revision = valid_revisions[0]["name"]

    warn(
        "Selected TinyTimeMixer revision "
        f"{selected_revision!r} for model_path={model_path!r}, "
        f"context_len={context_len}, horizon_len={horizon_len}."
    )

    return selected_revision


def _groupby_instance_levels(data):
    levels = list(range(data.index.nlevels - 1))
    return levels[0] if len(levels) == 1 else levels


def _same_index(data):
    level = _groupby_instance_levels(data)
    grouped = data.groupby(level=level, sort=False)

    indexes = grouped.apply(lambda x: x.index.get_level_values(-1))
    max_length = max(len(idx) for idx in indexes)
    return indexes.iloc[0], max_length


def _frame2numpy(data):
    _, length = _same_index(data)
    level = _groupby_instance_levels(data)
    groups = data.groupby(level=level, sort=False)

    arr = []
    for _, frame in groups:
        values = np.expand_dims(frame.values.astype(np.float32), axis=0)
        padded, _ = _pad_truncate(values, length)
        arr.append(padded[0])

    return np.stack(arr, axis=0)


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(
        self,
        y,
        context_length,
        prediction_length,
        X=None,
        frequency_token=None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        y : ndarray
            The time series data, shape (n_sequences, n_timestamps, n_dims)
        context_length : int
            The length of the past values
        prediction_length : int
            The length of the future values
        X : ndarray, optional (default=None)
            Exogenous variables, shape (n_sequences, n_timestamps, n_exog_dims)
            Should cover both past and future values for the entire time range
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.frequency_token = frequency_token

        # multi-index conversion for y
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        # Handle exogenous variables
        self.X = None
        if X is not None:
            if isinstance(X.index, pd.MultiIndex):
                self.X = _frame2numpy(X)
            else:
                self.X = np.expand_dims(X.values, axis=0)

        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = max(
            1, (self.n_timestamps - self.context_length - self.prediction_length + 1)
        )

    def __len__(self):
        """Return the length of the dataset."""
        # Calculate the number of samples that can be created from each sequence
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self.single_length
        n = i // self.single_length

        # Handle edge case where data is shorter than context + prediction length
        if self.n_timestamps < self.context_length + self.prediction_length:
            # Use all available data for past values and pad if necessary
            available_context = min(
                self.context_length, self.n_timestamps - self.prediction_length
            )
            available_context = max(1, available_context)

            past_values = self.y[n, :available_context, :]
            # Pad if context is shorter than required
            if past_values.shape[0] < self.context_length:
                padding_needed = self.context_length - past_values.shape[0]
                past_values = np.pad(
                    past_values,
                    ((padding_needed, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # For future values, take the last available data points
            future_start = max(0, self.n_timestamps - self.prediction_length)
            future_values = self.y[
                n, future_start : future_start + self.prediction_length, :
            ]

            # Pad future values if needed
            if future_values.shape[0] < self.prediction_length:
                padding_needed = self.prediction_length - future_values.shape[0]
                future_values = np.pad(
                    future_values,
                    ((0, padding_needed), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            past_values = self.y[n, m : m + self.context_length, :]
            future_values = self.y[
                n,
                m + self.context_length : m
                + self.context_length
                + self.prediction_length,
                :,
            ]

        observed_mask = np.ones_like(past_values)

        result = {
            "past_values": tensor(past_values).float(),
            "observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }

        # Add exogenous variables if available
        if self.X is not None:
            if self.n_timestamps < self.context_length + self.prediction_length:
                # Handle short sequences for exogenous variables
                available_context = min(
                    self.context_length, self.n_timestamps - self.prediction_length
                )
                available_context = max(1, available_context)

                past_exog = self.X[n, :available_context, :]
                if past_exog.shape[0] < self.context_length:
                    padding_needed = self.context_length - past_exog.shape[0]
                    past_exog = np.pad(
                        past_exog,
                        ((padding_needed, 0), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                future_start = max(0, self.n_timestamps - self.prediction_length)
                future_exog = self.X[
                    n, future_start : future_start + self.prediction_length, :
                ]

                if future_exog.shape[0] < self.prediction_length:
                    padding_needed = self.prediction_length - future_exog.shape[0]
                    future_exog = np.pad(
                        future_exog,
                        ((0, padding_needed), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
            else:
                # Normal case
                past_exog = self.X[n, m : m + self.context_length, :]
                future_exog = self.X[
                    n,
                    m + self.context_length : m
                    + self.context_length
                    + self.prediction_length,
                    :,
                ]

            # Concatenate past and future exogenous for the full sequence
            full_exog = np.concatenate([past_exog, future_exog], axis=0)
            result["exogenous_values"] = tensor(full_exog).float()

        if self.frequency_token is not None:
            result["freq_token"] = tensor(self.frequency_token).int()

        return result


@_multiton
class _CachedTinyTimeMixer:
    """Cached TinyTimeMixer model, ensuring one instance per loading config."""

    def __init__(
        self,
        key,
        model_path,
        revision,
        device,
        user_config,
        use_source_package,
        fit_strategy,
    ):
        self.key = key
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.user_config = user_config
        self.use_source_package = use_source_package
        self.fit_strategy = fit_strategy
        self.model_ = None
        self._ttm_classes = None

    def load(self):
        """Load and return the cached model."""
        if self.model_ is not None:
            return self.model_

        config = self._build_config()
        self.model_, info = self._load_model(config)
        self.model_ = self.model_.to(self.device)
        self._set_training_parameters(info)

        return self.model_

    def _set_training_parameters(self, info):
        """Set trainable parameters based on fit strategy."""
        mismatched_keys = info["mismatched_keys"]

        if self.fit_strategy == "full":
            self._set_requires_grad(True)
            return

        if self.fit_strategy == "minimal":
            self._set_requires_grad(False)
            for key in mismatched_keys:
                self._set_mismatched_key_trainable(key)
            return

        if self.fit_strategy == "zero-shot":
            if len(mismatched_keys) > 0:
                raise ValueError(
                    "fit_strategy='zero-shot' requires all pretrained weights to "
                    "match the loaded model configuration, but mismatched weights "
                    f"were found: {self._format_mismatched_keys(mismatched_keys)}. "
                    "Use fit_strategy='minimal' or 'full', or choose a compatible "
                    "model/configuration."
                )
            self._set_requires_grad(False)
            return

        raise ValueError(
            "Unknown fit_strategy. Expected one of 'zero-shot', 'minimal', "
            f"or 'full', but found {self.fit_strategy!r}."
        )

    def _set_requires_grad(self, requires_grad):
        """Set trainability for all model parameters."""
        for param in self.model_.parameters():
            param.requires_grad = requires_grad

    def _set_mismatched_key_trainable(self, key):
        """Set the module parameter corresponding to a mismatched key trainable."""
        # transformers>=5.0 may return tuples (key, shape) instead of plain strings.
        if isinstance(key, tuple):
            key = key[0]

        module = self.model_
        for attr_name in key.split(".")[:-1]:
            module = getattr(module, attr_name)

        parameter_name = key.split(".")[-1]
        getattr(module, parameter_name).requires_grad = True

    @staticmethod
    def _format_mismatched_keys(mismatched_keys):
        """Format mismatched key names for error messages."""
        formatted_keys = []
        for key in mismatched_keys:
            if isinstance(key, tuple):
                key = key[0]
            formatted_keys.append(str(key))
        return ", ".join(formatted_keys)

    def _load_model(self, config):
        """Load pretrained weights or initialize from config."""
        if self.model_path is not None:
            return self._load_from_pretrained(config)
        return self._load_from_config(config)

    def _load_from_pretrained(self, config):
        """Load TinyTimeMixer weights from ``self.model_path``."""
        TinyTimeMixerModel = self._get_model_class(config)

        return TinyTimeMixerModel.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )

    def _load_from_config(self, config):
        """Initialize TinyTimeMixer from config with random weights."""
        TinyTimeMixerModel = self._get_model_class(config)

        model = TinyTimeMixerModel(config=config)
        info = {"mismatched_keys": []}
        return model, info

    def _build_config(self):
        """Build the effective TinyTimeMixer config used for loading."""
        config = self._load_base_config()
        config_dict = config.to_dict()
        config_dict.update(self.user_config)
        return config.from_dict(config_dict)

    def _load_base_config(self):
        """Load pretrained config or create a default config for random init."""
        TinyTimeMixerConfig, _, _ = self._get_ttm_classes()

        if self.model_path is not None:
            return TinyTimeMixerConfig.from_pretrained(
                self.model_path,
                revision=self.revision,
            )

        if self.fit_strategy != "full":
            raise ValueError(
                "Invalid configuration: 'model_path' is set to None."
                "This requires 'fit_strategy' to be 'full'."
                "Please set 'fit_strategy' to 'full' or provide a valid model path."
            )

        config = TinyTimeMixerConfig()
        # call to initialize attributes like num_patchess
        config.check_and_init_preprocessing()
        return config

    def _get_model_class(self, config):
        """Return the TinyTimeMixer model class matching ``config``."""
        _, TinyTimeMixerForPrediction, TinyTimeMixerForDecomposedPrediction = (
            self._get_ttm_classes()
        )

        if self._is_decomposed_config(config):
            return TinyTimeMixerForDecomposedPrediction

        return TinyTimeMixerForPrediction

    @staticmethod
    def _is_decomposed_config(config):
        """Return whether ``config`` describes a decomposed TTM model."""
        return any(
            getattr(config, attr, None) is not None
            for attr in (
                "residual_context_length",
                "trend_patch_length",
                "trend_patch_stride",
                "trend_d_model",
                "trend_decoder_d_model",
                "trend_num_layers",
                "trend_decoder_num_layers",
                "trend_register_tokens",
                "trend_fft_length",
                "trend_multi_scale",
                "trend_adaptive_patching_levels",
                "trend_head_d_model",
            )
        )

    def _get_ttm_classes(self):
        """Return TinyTimeMixer config/model classes."""
        if self._ttm_classes is not None:
            return self._ttm_classes

        if self.use_source_package:
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerConfig,
                TinyTimeMixerForDecomposedPrediction,
                TinyTimeMixerForPrediction,
            )
        else:
            from sktime.libs.granite_ttm import TinyTimeMixerConfig
            from sktime.libs.granite_ttm.modeling_tinytimemixer import (
                TinyTimeMixerForDecomposedPrediction,
                TinyTimeMixerForPrediction,
            )

        self._ttm_classes = (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
            TinyTimeMixerForDecomposedPrediction,
        )
        return self._ttm_classes
