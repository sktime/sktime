# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the TSPulse time series classifier via granite-tsfm."""

__author__ = ["Faakhir30"]
__all__ = ["TSPulseClassifier"]

import tempfile

from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")

_DEFAULT_MODEL_CONFIG = {
    "head_gated_attention_activation": "softmax",
    "channel_virtual_expand_scale": 2,
    "mask_ratio": 0.3,
    "head_reduce_d_model": 1,
    "disable_mask_in_classification_eval": True,
    "fft_time_consistent_masking": True,
    "decoder_mode": "mix_channel",
    "head_aggregation_dim": "patch",
    "head_aggregation": None,
    "loss": "cross_entropy",
    "ignore_mismatched_sizes": True,
}

LABEL_COLUMN = "__TS_PULSE_LABEL_COLUMN__"


class TSPulseClassifier(BaseClassifier):
    """Time series classifier wrapping IBM TSPulse via granite-tsfm.

    Loads a pretrained TSPulse backbone from Hugging Face, freezes most backbone
    weights, and fine-tunes the classification head and patch-embedding layers
    on the training data.

    Implementation adapted from [1].

    Parameters
    ----------
    model_path : str, default="ibm-granite/granite-timeseries-tspulse-r1"
        Hugging Face model id or local path.
    revision : str, default="tspulse-block-dualhead-512-p16-r1"
        Model revision on the Hugging Face Hub.
    config : dict, optional, default=None
        Extra keyword arguments forwarded to
        ``tsfm_public.models.tspulse.TSPulseForClassification.from_pretrained``.

        Any key you supply overrides the built-in default for that key, *except*
        for the following two keys which are always inferred from the training
        data:

        - ``num_input_channels``: set to the number of channels in ``X``
        - ``num_targets``: set to the number of distinct labels in ``y``

        If ``config`` is ``None``, then following default overrides are applied:

        - ``head_gated_attention_activation="softmax"``:
          gated attention activation in the classification head
        - ``channel_virtual_expand_scale=2``:
          virtual expansion factor for channel mixing in the decoder/head
        - ``mask_ratio=0.3``:
          fraction of patches masked during training (use ``0`` to disable)
        - ``head_reduce_d_model=1``:
          reduction factor for model dimension in the head
        - ``disable_mask_in_classification_eval=True``:
          disables masking at evaluation/prediction time
        - ``fft_time_consistent_masking=True``:
          uses masked time-series for FFT during training
        - ``decoder_mode="mix_channel"``:
          decoder channel mixing mode (alternative: ``"common_channel"``)
        - ``head_aggregation_dim="patch"``:
          aggregation dimension used by the head
        - ``head_aggregation=None``:
          use model-default aggregation module
        - ``loss="cross_entropy"``:
          loss for classification fine-tuning
        - ``ignore_mismatched_sizes=True``:
          allows loading when head shapes differ from the checkpoint

    context_length : int, default=512
        Series length passed to the preprocessor and dataset.
    scaling : bool, default=True
        Whether to scale input channels in the preprocessor.
    batch_size : int, default=32
        Training batch size.
    epochs : int, default=1
        Number of fine-tuning epochs.
    learning_rate : float, optional, default=1e-3
        Optimizer learning rate for fine-tuning.
    freeze_backbone : bool, default=True
        If True, freeze backbone weights except patch-embedding layers.
    train_val_split : float, default=0.0
        Fraction of training data held out for validation during fine-tuning.
        ``0.0`` uses all training data.
    device : str, default="auto"
        PyTorch device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
    seed : int, default=42
        Random seed for training.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/tspulse_classification.ipynb
    .. [2] https://arxiv.org/abs/2505.13033

    Examples
    --------
    >>> from sktime.classification.foundation_models.tspulse import TSPulseClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_type="nested_univ")
    >>> X_test, _ = load_unit_test(split="test", return_type="nested_univ")
    >>> clf = TSPulseClassifier(epochs=1, batch_size=8)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "wgifford",
            "ajati",
            "subodh2702",
            "vijaye12ibm",
            "summukhe",
            "Tomoya Sakai",
            "Pankaj Dayama",
            "Jayant Kalagnanam",
            "Faakhir30",
        ],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.11",
        "python_dependencies": [
            "granite-tsfm>=0.3.5",
            "torch",
            "transformers",
            "accelerate",
        ],
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "serialization:native_artifacts": ("_model",),
        "serialization:skip": ("_pipeline",),
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-block-dualhead-512-p16-r1",
        config=None,
        context_length=512,
        scaling=True,
        batch_size=32,
        epochs=1,
        learning_rate=1e-3,
        freeze_backbone=True,
        train_val_split=0.0,
        device="auto",
        seed=42,
    ):
        self.model_path = model_path
        self.revision = revision
        self.config = config
        self.context_length = context_length
        self.scaling = scaling
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.train_val_split = train_val_split
        self.device = device
        self.seed = seed
        super().__init__()

    def _fit(self, X, y):
        f"""Fit the TSPulse classifier to the training data.

        Column name ``{LABEL_COLUMN}`` is reserved for internal use.
        Please avoid having column with this name in ``X``.

        Parameters
        ----------
        X : pd.DataFrame (nested_univ)
            Panel data in nested format (cells are ``pd.Series``).
        y : np.ndarray, optional
            Class labels aligned with rows of ``X``.
        """
        from torch.utils.data import random_split
        from transformers import Trainer, TrainingArguments, set_seed
        from tsfm_public.models.tspulse import TSPulseForClassification
        from tsfm_public.toolkit.dataset import ClassificationDFDataset
        from tsfm_public.toolkit.time_series_classification_pipeline import (
            TimeSeriesClassificationPipeline,
        )
        from tsfm_public.toolkit.time_series_classification_preprocessor import (
            TimeSeriesClassificationPreprocessor,
        )

        set_seed(self.seed)
        self._y_dtype = y.dtype
        self._device = _resolve_device(self.device)

        df = X.copy()
        # tsfm_public classification preprocessor expects integer-like row indices
        # (it uses them to build repeated indices after unnesting).
        df = df.reset_index(drop=True)
        if y is not None:
            df[LABEL_COLUMN] = y
        input_columns = [c for c in X.columns if c != LABEL_COLUMN]

        preprocessor = TimeSeriesClassificationPreprocessor(
            input_columns=input_columns,
            label_column=LABEL_COLUMN,
            scaling=self.scaling,
            context_length=self.context_length,
        )
        preprocessor.train(df)
        df_prep = preprocessor.preprocess(df)
        self._preprocessor = preprocessor

        model_config = _build_model_config(
            preprocessor=preprocessor,
            df=df,
            user_config=self.config,
        )
        model = TSPulseForClassification.from_pretrained(
            self.model_path,
            revision=self.revision,
            **model_config,
        )
        if self.freeze_backbone:
            _freeze_backbone(model)
        model = model.to(self._device)

        dataset = ClassificationDFDataset(
            df_prep,
            input_columns=input_columns,
            label_column=LABEL_COLUMN,
            context_length=self.context_length,
            enable_padding=False,
            full_series=True,
        )

        eval_dataset = None
        if 0 < self.train_val_split < 1:
            n_val = max(1, int(len(dataset) * self.train_val_split))
            n_train = len(dataset) - n_val
            train_dataset, eval_dataset = random_split(dataset, [n_train, n_val])
        else:
            train_dataset = dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_args = {
                "output_dir": tmp_dir,
                "overwrite_output_dir": True,
                "learning_rate": self.learning_rate,
                "num_train_epochs": self.epochs,
                "per_device_train_batch_size": self.batch_size,
                "per_device_eval_batch_size": self.batch_size,
                "eval_strategy": "epoch" if eval_dataset is not None else "no",
                "save_strategy": "no",
                "logging_strategy": "no",
                "report_to": "none",
                "seed": self.seed,
            }
            if self._device == "cpu":
                train_args["dataloader_pin_memory"] = False
            args = TrainingArguments(**train_args)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()

        model.eval()
        self._model = model
        self._pipeline = TimeSeriesClassificationPipeline(
            model,
            feature_extractor=preprocessor,
            device=self._device,
        )
        return self

    def _predict(self, X):
        f"""Predict labels for the test data.

        Column name ``{LABEL_COLUMN}`` is reserved for internal use.
        Please avoid having column with this name in ``X``.
        """
        if not hasattr(self, "_pipeline") or self._pipeline is None:
            from tsfm_public.toolkit.time_series_classification_pipeline import (
                TimeSeriesClassificationPipeline,
            )

            self._pipeline = TimeSeriesClassificationPipeline(
                self._model,
                feature_extractor=self._preprocessor,
                device=self._device,
            )

        df = X.copy()
        df = df.reset_index(drop=True)
        if LABEL_COLUMN not in df.columns:
            df[LABEL_COLUMN] = self.classes_[0]

        self._model.eval()
        out = self._pipeline(df)
        pred_col = f"{LABEL_COLUMN}_prediction"
        preds = out[pred_col].to_numpy()
        return preds.astype(self._y_dtype)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        return [
            {"epochs": 1, "batch_size": 8, "train_val_split": 0.0},
            {"epochs": 1, "batch_size": 16, "freeze_backbone": False},
        ]


def _build_model_config(preprocessor, df, user_config):
    config = _DEFAULT_MODEL_CONFIG.copy()
    if user_config is not None:
        config.update(user_config)
    config["num_input_channels"] = preprocessor.num_input_channels
    config["num_targets"] = df[LABEL_COLUMN].nunique()
    return config


def _freeze_backbone(model):
    """Freeze backbone; keep patch-embedding layers trainable."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.time_encoding.parameters():
        param.requires_grad = True
    for param in model.backbone.fft_encoding.parameters():
        param.requires_grad = True


def _resolve_device(device):
    if device in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"

    # device == "auto"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
