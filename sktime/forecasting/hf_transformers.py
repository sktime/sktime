"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["benheid", "geetu040"]


class HFTransformersForecaster(BaseForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    This forecaster fetches the model from the huggingface model hub.
    Note, this forecaster is in an experimental state. It is currently only
    working for Informer, Autoformer, and TimeSeriesTransformer.

    Parameters
    ----------
    model_path : str or PreTrainedModel
        Path to the huggingface model to use for forecasting. Currently,
        Informer, Autoformer, and TimeSeriesTransformer are supported.
        This can be one of the following:
        - A string specifying the Hugging Face model name or path
          (e.g., `"huggingface/autoformer-tourism-monthly"`).
        - An instance of a `PreTrainedModel`, allowing manual initialization
          and configuration.
    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:

        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time.
        - "peft": Applies Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt
          the model with fewer trainable parameters, saving computational resources.

          Note: If the 'peft' package is not available, a `ModuleNotFoundError` will
          be raised, indicating that the 'peft' package is required. Please install
          it using `pip install peft` to use this fit strategy.

    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model. Configuration objects inherit from
        ``PreTrainedConfig`` and can be used to control the model outputs and
        architecture. Refer to the individual model config for particular
        model-specific config params.

        ``PreTrainedConfig`` is the base class for all configuration classes.
        It handles a few parameters common to all models' configurations as
        well as methods for loading/downloading/saving configurations. A
        configuration file can be loaded and saved to disk. Loading the
        configuration file and using this file to initialize a model does not
        load the model weights. It only affects the model's configuration.

        Keys supported by ``PreTrainedConfig`` include:

        name_or_path : str, optional, default=""
            Store the string that was passed to
            ``PreTrainedModel.from_pretrained()`` as
            ``pretrained_model_name_or_path`` if the configuration was created
            with such a method.
        output_hidden_states : bool, optional, default=False
            Whether or not the model should return all hidden-states.
        output_attentions : bool, optional, default=False
            Whether or not the model should return all attentions.
        return_dict : bool, optional, default=True
            Whether or not the model should return a ``ModelOutput`` instead of
            a plain tuple.
        is_encoder_decoder : bool, optional, default=False
            Whether the model is used as an encoder/decoder or not.
        chunk_size_feed_forward : int, optional, default=0
            The chunk size of all feed forward layers in the residual attention
            blocks. A chunk size of ``0`` means that the feed forward layer is
            not chunked. A chunk size of ``n`` means that the feed forward layer
            processes ``n < sequence_length`` embeddings at a time.
        per_layer_config : dict, optional
            A sparse mapping from layer indices to configuration attribute
            overrides. Each key is a layer index, and each value contains the
            attributes that differ from the global config for that layer.

        Parameters for fine-tuning tasks:

        architectures : list of str, optional
            Model architectures that can be used with the model pretrained
            weights.
        id2label : dict of int to str, optional
            A map from index (for instance prediction index, or target index)
            to label.
        label2id : dict of str to int, optional
            A map from label to index for the model.
        num_labels : int, optional
            Number of labels to use in the last layer added to the model,
            typically for a classification task.
        problem_type : str, optional
            Problem type for ``XxxForSequenceClassification`` models. Can be
            one of ``"regression"``, ``"single_label_classification"`` or
            ``"multi_label_classification"``.

        PyTorch specific parameters:

        dtype : str, optional
            The dtype of the weights. This attribute can be used to initialize
            the model to a non-default dtype (which is normally ``float32``)
            and thus allow for optimal storage allocation. For example, if the
            saved model is ``float16``, ideally we want to load it back using
            the minimal amount of memory needed to load ``float16`` weights.

        Class attributes (overridden by derived classes):

        model_type : str
            An identifier for the model type, serialized into the JSON file,
            and used to recreate the correct object in ``AutoConfig``.
        has_no_defaults_at_init : bool
            Whether the config class can be initialized without providing input
            arguments. Some configurations require inputs to be defined at init
            and have no default values, usually these are composite configs
            (but not necessarily) such as ``EncoderDecoderConfig`` or
            ``RagConfig``. They have to be initialized from two or more configs
            of type ``PreTrainedConfig``.
        keys_to_ignore_at_inference : list of str
            A list of keys to ignore by default when looking at dictionary
            outputs of the model during inference.
        attribute_map : dict of str to str
            A dict that maps model specific attribute names to the standardized
            naming of attributes.
        base_model_tp_plan : dict
            A dict that maps sub-modules FQNs of a base model to a tensor
            parallel plan applied to the sub-module when
            ``model.tensor_parallel`` is called.
        base_model_fsdp_plan : dict
            A dict that maps sub-modules of a base model to an FSDP2 sharding
            strategy (e.g. ``"free_full_weight"`` / ``"keep_full_weight"``).
            Keys can be wildcard module paths (e.g. ``"layers.*"``) or tuples
            of paths (grouped into a single ``fully_shard`` call).
        base_model_pp_plan : dict of str to tuple of list of str
            A dict that maps child-modules of a base model to a pipeline
            parallel plan that enables users to place the child-module on the
            appropriate device.

        Common attributes (present in all subclasses):

        vocab_size : int
            The number of tokens in the vocabulary, which is also the first
            dimension of the embeddings matrix (this attribute may be missing
            for models that don't have a text modality like ViT).
        hidden_size : int
            The hidden size of the model.
        num_attention_heads : int
            The number of attention heads used in the multi-head attention
            layers of the model.
        num_hidden_layers : int
            The number of blocks in the model.

    training_args : dict, default={}
        Training arguments to use for the model. See
        ``transformers.TrainingArguments`` for details [1]_.
        Note that the ``output_dir`` argument is required.
    compute_metrics : list, default=None
        List of metrics to compute during training. See ``transformers.Trainer``
        for details.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training. See ``transformers.Trainer``
    peft_config : peft.PeftConfig, default=None
        Configuration for Parameter-Efficient Fine-Tuning.
        When ``fit_strategy`` is set to "peft",
        this will be used to set up PEFT parameters for the model.
        See the ``peft`` documentation for details [2]_.

    References
    ----------
    .. [1] https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments
    .. [2] https://huggingface.co/docs/peft/en/package_reference/config#peft.PeftConfig

    Examples
    --------
    **Using a Pretrained Model from Hugging Face**

    >>> from sktime.forecasting.hf_transformers import HFTransformersForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    training_args ={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    },
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    },
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP

    **Using PEFT for Fine-Tuning**

    >>> from sktime.forecasting.hf_transformers import HFTransformersForecaster
    >>> from sktime.datasets import load_airline # doctest: +SKIP
    >>> from peft import LoraConfig # doctest: +SKIP
    >>> y = load_airline() # doctest: +SKIP
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    fit_strategy="peft",
    ...    training_args={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    }, # doctest: +SKIP
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    }, # doctest: +SKIP
    ...    peft_config=LoraConfig(
    ...        r=8,
    ...        lora_alpha=32,
    ...        target_modules=["q_proj", "v_proj"],
    ...        lora_dropout=0.01,
    ...    ) # doctest: +SKIP
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP

    **Using an Initialized Model**

    >>> from sktime.datasets import load_airline
    >>> from transformers import AutoformerConfig, AutoformerForPrediction
    >>> from sktime.forecasting.hf_transformers import HFTransformersForecaster
    >>> y = load_airline()

    >>> # Define model configuration
    >>> config = AutoformerConfig(
    ...     num_dynamic_real_features=0,
    ...     num_static_real_features=0,
    ...     num_static_categorical_features=0,
    ...     num_time_features=0,
    ...     context_length=32,
    ...     prediction_length=8,
    ...     lags_sequence=[1, 2, 3],
    ... )

    >>> # Initialize the model
    >>> model = AutoformerForPrediction(config)

    >>> # Initialize the forecaster with the model
    >>> forecaster = HFTransformersForecaster(
    ...     model_path=model,
    ...     fit_strategy="minimal",
    ...     training_args={
    ...         "num_train_epochs": 10,
    ...         "output_dir": "output",
    ...         "per_device_train_batch_size": 4
    ...     },
    ... )

    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> fh = [1, 2, 3] # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["benheid", "geetu040"],
        "maintainers": ["benheid", "geetu040"],
        # estimator type
        # --------------
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:unequal_length": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_hf_transformers_forecaster"],
        "tests:python_dependencies": ["peft"],
    }

    def __init__(
        self,
        model_path: str = None,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
        peft_config=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.validation_split = validation_split
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self.deterministic = deterministic
        self.callbacks = callbacks
        self._callbacks = callbacks
        self.peft_config = peft_config

    def _fit(self, y, X, fh):
        from transformers import AutoConfig, PreTrainedModel, Trainer, TrainingArguments

        if isinstance(self.model_path, PreTrainedModel):
            self.model = self.model_path
            self.info = {"mismatched_keys": []}
            config = self.model.config

        else:
            # Load model and extract config
            config = AutoConfig.from_pretrained(self.model_path)

            # Update config with user-provided config
            _config = config.to_dict()
            _config.update(self._config)
            _config["num_dynamic_real_features"] = 0
            _config["num_static_real_features"] = 0
            _config["num_static_categorical_features"] = 0
            _config["num_time_features"] = 0 if X is None else X.shape[-1]

            if hasattr(config, "feature_size"):
                del _config["feature_size"]

            if fh is not None:
                _config["prediction_length"] = max(
                    *fh.to_relative(self._cutoff)._values,
                    _config.get("prediction_length", 0),
                )

            config = config.from_dict(_config)

            # Load model and info
            import transformers

            prediction_model_class = None
            if hasattr(config, "architectures") and config.architectures:
                prediction_model_class = config.architectures[0]
            elif hasattr(config, "model_type"):
                prediction_model_class = (
                    "".join(x.capitalize() for x in config.model_type.split("_"))
                    + "ForPrediction"
                )
            else:
                raise ValueError("The model type cannot be inferred from the config.")

            self.model, self.info = getattr(
                transformers, prediction_model_class
            ).from_pretrained(
                self.model_path,
                config=config,
                output_loading_info=True,
                ignore_mismatched_sizes=True,
            )

            # Freeze loaded parameters and reinitialize mismatched layers
            for param in self.model.parameters():
                param.requires_grad = False
            for key, _, _ in self.info["mismatched_keys"]:
                _model = self.model
                for attr_name in key.split(".")[:-1]:
                    _model = getattr(_model, attr_name)
                _model.weight = torch.nn.Parameter(
                    _model.weight.masked_fill(_model.weight.isnan(), 0.001),
                    requires_grad=True,
                )

        # Dataset preparation
        if self.validation_split is not None:
            split = int(len(y) * (1 - self.validation_split))
            train_dataset = PyTorchDataset(
                y[:split],
                config.context_length + max(config.lags_sequence),
                X=X[:split] if X is not None else None,
                fh=config.prediction_length,
            )
            eval_dataset = PyTorchDataset(
                y[split:],
                config.context_length + max(config.lags_sequence),
                X=X[split:] if X is not None else None,
                fh=config.prediction_length,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                config.context_length + max(config.lags_sequence),
                X=X if X is not None else None,
                fh=config.prediction_length,
            )
            eval_dataset = None

        # Prepare training arguments
        training_args = deepcopy(self.training_args)
        training_args["label_names"] = ["future_values"]
        # evaluation_strategy was renamed to eval_strategy in transformers 4.41.0
        if _check_soft_dependencies("transformers>=4.41.0", severity="none"):
            if "evaluation_strategy" in training_args:
                training_args["eval_strategy"] = training_args.pop(
                    "evaluation_strategy"
                )
        training_args = TrainingArguments(**training_args)

        # Handle fine-tuning strategy
        if self.fit_strategy == "minimal":
            if not any(param.requires_grad for param in self.model.parameters()):
                return
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.fit_strategy == "peft":
            if _check_soft_dependencies(
                "peft",
                severity="error",
                msg=(
                    f"Error in {self.__class__.__name__}: 'peft' module not found. "
                    "'peft' is a soft dependency and not included "
                    "in the base sktime installation. "
                    "To use this functionality, please install 'peft' by running: "
                    "`pip install peft` or `pip install sktime[dl]`. "
                    "To install all soft dependencies, "
                    "run: `pip install sktime[all_extras]`"
                ),
            ):
                from peft import get_peft_model

            peft_config = deepcopy(self.peft_config)
            self.model = get_peft_model(self.model, peft_config)
        else:
            raise ValueError("Unknown fit strategy")

        # Train the model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=self._callbacks,
        )
        trainer.train()

    def _predict(self, fh, X=None):
        import transformers

        if self.deterministic:
            transformers.set_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.eval()
        from torch import from_numpy

        hist = self._y.values.reshape((1, -1))
        if X is not None:
            hist_x = self._X.values.reshape((1, -1, self._X.shape[-1]))
            x_ = X.values.reshape((1, -1, self._X.shape[-1]))
            if x_.shape[1] < self.model.config.prediction_length:
                # TODO raise exception here?
                x_ = np.resize(
                    x_, (1, self.model.config.prediction_length, x_.shape[-1])
                )
        else:
            hist_x = np.array(
                [
                    [[]]
                    * (
                        self.model.config.context_length
                        + max(self.model.config.lags_sequence)
                    )
                ]
            )
            x_ = np.array([[[]] * self.model.config.prediction_length])

        pred = self.model.generate(
            past_values=from_numpy(hist).to(self.model.dtype).to(self.model.device),
            past_time_features=from_numpy(
                hist_x[
                    :,
                    -self.model.config.context_length
                    - max(self.model.config.lags_sequence) :,
                ]
            )
            .to(self.model.dtype)
            .to(self.model.device),
            future_time_features=from_numpy(x_)
            .to(self.model.dtype)
            .to(self.model.device),
            past_observed_mask=from_numpy((~np.isnan(hist)).astype(int)).to(
                self.model.device
            ),
        )

        pred = pred.sequences.mean(dim=1).detach().cpu().numpy().T

        pred = pd.Series(
            pred.reshape((-1,)),
            index=ForecastingHorizon(range(1, len(pred) + 1))
            .to_absolute(self._cutoff)
            ._values,
            # columns=self._y.columns
            name=self._y.name,
        )
        return pred.loc[fh.to_absolute(self.cutoff)._values]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict contains parameters to construct an "interesting" test instance,
            i.e., `MyClass(**params)` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        base_training_args = {
            "num_train_epochs": 1,
            "output_dir": "test_output",
            "per_device_train_batch_size": 32,
        }

        base_config = {
            "lags_sequence": [1, 2, 3],
            "context_length": 2,
            "prediction_length": 4,
        }

        test_params = [
            # General transformer-based test cases
            {
                "model_path": "huggingface/informer-tourism-monthly",
                "fit_strategy": "minimal",
                "training_args": base_training_args,
                "config": base_config,
                "deterministic": True,
            }
        ]

        # Add PEFT-specific test case if PEFT is available
        if _check_soft_dependencies("peft", severity="none"):
            from peft import LoraConfig

            peft_test_case = {
                "model_path": "huggingface/autoformer-tourism-monthly",
                "fit_strategy": "peft",
                "training_args": base_training_args,
                "config": {**base_config, "label_length": 2},
                "peft_config": LoraConfig(
                    r=2,
                    lora_alpha=8,
                    target_modules=["q_proj"],
                    lora_dropout=0.01,
                ),
                "deterministic": True,
            }
            test_params.append(peft_test_case)

        return test_params


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh=None, X=None):
        self.y = y.values
        self.X = X.values if X is not None else X
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return max(len(self.y) - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        hist_y = tensor(self.y[i : i + self.seq_len]).float()
        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
            hist_exog = tensor(self.X[i : i + self.seq_len]).float()
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)
        return {
            "past_values": hist_y,
            "past_time_features": hist_exog,
            "future_time_features": exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values": from_numpy(
                self.y[i + self.seq_len : i + self.seq_len + self.fh]
            ).float(),
        }
