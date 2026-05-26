# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the TSPulse time series classifier via granite-tsfm."""

__author__ = ["Faakhir30"]
__all__ = ["TSPulseClassifier"]

import tempfile

import pandas as pd

from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_numpy3d_to_dflist
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
        Keyword arguments passed to the Hugging Face loader when building the
        classification model. Any key you supply overrides the built-in default
        for that key. Two keys are always set from the training data and cannot
        be overridden: ``num_input_channels`` (number of input series channels)
        and ``num_targets`` (number of distinct classes).

        If ``config`` is ``None``, the following defaults are used (from the
        granite-tsfm classification tutorial):

        ``head_gated_attention_activation`` : str, default ``"softmax"``
            Activation for the gated attention in the classification head.
            Alternatives include ``"sigmoid"``.
        ``channel_virtual_expand_scale`` : int, default ``2``
            Expansion factor for the channel-mixing block in the decoder head.
        ``mask_ratio`` : float, default ``0.3``
            Fraction of patches masked during training (masked modelling).
            Use ``0`` to disable masking.
        ``head_reduce_d_model`` : int, default ``1``
            Reduction factor applied to the model dimension in the head.
        ``disable_mask_in_classification_eval`` : bool, default ``True``
            If ``True``, masking is turned off at evaluation and prediction time.
        ``fft_time_consistent_masking`` : bool, default ``True``
            If ``True``, apply FFT-based time-consistent masking during training.
        ``decoder_mode`` : str, default ``"mix_channel"``
            How channels are combined in the decoder. Alternative:
            ``"common_channel"``.
        ``head_aggregation_dim`` : str, default ``"patch"``
            Dimension along which the classification head aggregates features.
        ``head_aggregation`` : None, default ``None``
            Optional aggregation module for the head; ``None`` uses the model default.
        ``loss`` : str, default ``"cross_entropy"``
            Training loss used by the model.
        ``ignore_mismatched_sizes`` : bool, default ``True``
            If ``True``, allow loading pretrained weights even when the
            classification head shape differs from the checkpoint (head weights
            are then randomly initialized).
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
    label_column : str, default="class_vals"
        Label column name used in the granite-tsfm preprocessor.
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
    >>> X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    >>> X_test, _ = load_unit_test(split="test", return_type="numpy3d")
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
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
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
        label_column="class_vals",
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
        self.label_column = label_column
        self.seed = seed
        super().__init__()

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable model pipeline."""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_pipeline"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _fit(self, X, y):
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

        df = _panel_to_nested_df(X, y, label_column=self.label_column)
        self._input_columns = [f"dim_{j}" for j in range(X.shape[1])]

        preprocessor = TimeSeriesClassificationPreprocessor(
            input_columns=self._input_columns,
            label_column=self.label_column,
            scaling=self.scaling,
            context_length=self.context_length,
        )
        preprocessor.train(df)
        df_prep = preprocessor.preprocess(df)
        self._preprocessor = preprocessor

        model_config = _build_model_config(
            preprocessor=preprocessor,
            df=df,
            label_column=self.label_column,
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
            input_columns=self._input_columns,
            label_column=self.label_column,
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

        self._model = model
        self._pipeline = TimeSeriesClassificationPipeline(
            model,
            feature_extractor=preprocessor,
            device=self._device,
        )
        return self

    def _predict(self, X):
        df = _panel_to_nested_df(X, label_column=self.label_column)
        if self.label_column not in df.columns:
            df[self.label_column] = self.classes_[0]

        out = self._pipeline(df)
        pred_col = f"{self.label_column}_prediction"
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


def _panel_to_nested_df(X, y=None, label_column="class_vals"):
    """Convert numpy3D panel data to nested DataFrame for granite-tsfm."""
    dflist = from_numpy3d_to_dflist(X)
    n_vars = X.shape[1]
    data = {
        f"dim_{j}": [dflist[i][f"var_{j}"] for i in range(len(dflist))]
        for j in range(n_vars)
    }
    if y is not None:
        data[label_column] = y
    return pd.DataFrame(data)


def _build_model_config(preprocessor, df, label_column, user_config):
    config = _DEFAULT_MODEL_CONFIG.copy()
    if user_config is not None:
        config.update(user_config)
    config["num_input_channels"] = preprocessor.num_input_channels
    config["num_targets"] = df[label_column].nunique()
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
