"""Interface for the momentfm deep learning time series classifier."""

from copy import deepcopy

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.classification.base import BaseClassifier
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

accelerate = _safe_import("accelerate")
CrossEntropyLoss = _safe_import("torch.nn.CrossEntropyLoss")
Adam = _safe_import("torch.optim.Adam")
OneCycleLR = _safe_import("torch.optim.lr_scheduler.OneCycleLR")
DataLoader = _safe_import("torch.utils.data.DataLoader")
Dataset = _safe_import("torch.utils.data.Dataset")
empty_cache = _safe_import("torch.cuda.empty_cache")

if _check_soft_dependencies("transformers", severity="none"):
    from sktime.libs.momentfm import MOMENTPipeline


def _momentfm_cache_key(
    pretrained_model_name_or_path, n_channels, num_class, dropout, device
):
    """Create a deterministic cache key for a MomentFM model configuration.

    Parameters
    ----------
    pretrained_model_name_or_path : str
        HuggingFace model identifier or local path.
    n_channels : int
        Number of input channels.
    num_class : int
        Number of output classes.
    dropout : float
        Dropout probability for the classification head.
    device : str
        Device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    str
        A string key that uniquely identifies this model configuration.
    """
    parts = [
        str(pretrained_model_name_or_path),
        str(n_channels),
        str(num_class),
        str(dropout),
        str(device),
    ]
    return "_".join(parts)


@_multiton
class _CachedMomentFMPipeline:
    """Cached MOMENTPipeline loader; ensures one backbone per unique configuration.

    MomentFM loads large pretrained weights from HuggingFace.  By wrapping the
    pipeline in a multiton, successive ``fit()`` calls with identical parameters
    skip the expensive ``from_pretrained()`` download/load and reuse the already-
    initialised pipeline instead.

    Parameters
    ----------
    key : str
        Unique cache key produced by :func:`_momentfm_cache_key`.
    pretrained_model_name_or_path : str
        HuggingFace model identifier or local path.
    n_channels : int
        Number of input channels.
    num_class : int
        Number of output classes.
    dropout : float
        Dropout probability for the classification head.
    device : str
        Device string used by the pipeline.
    """

    def __init__(
        self,
        key,
        pretrained_model_name_or_path,
        n_channels,
        num_class,
        dropout,
        device,
    ):
        self.key = key
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.n_channels = n_channels
        self.num_class = num_class
        self.dropout = dropout
        self.device = device
        self._pipeline = None

    def load(self):
        """Load (or return cached) MOMENTPipeline for this configuration.

        Returns
        -------
        MOMENTPipeline
            The initialised pipeline, ready for fine-tuning.
        """
        if self._pipeline is None:
            self._pipeline = MOMENTPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                model_kwargs={
                    "task_name": "classification",
                    "n_channels": self.n_channels,
                    "num_class": self.num_class,
                    "dropout": self.dropout,
                    "device": self.device,
                },
            )
            self._pipeline.init()
        return self._pipeline


class MomentFMClassifier(BaseClassifier):
    """
    Interface for classification with the deep learning time series model momentfm.

    MomentFM is a collection of open source foundation models for the general
    purpose of time series analysis. The Moment Foundation Model is a pre-trained
    model that is capable of accomplishing various time series tasks, such as:
        - Classification

    This interface with MomentFM focuses on the classification task, in which the
    foundation model uses a user fine tuned 'classification head' to classify a time
    series. This model does NOT have zero shot capabilities and requires fine-tuning
    to achieve performance on user inputted data.

    NOTE: This model can only handle time series with a sequence length of 512 or less.

    For more information: see
    https://github.com/moment-timeseries-foundation-model/moment

    For information regarding licensing and use of the momentfm model please visit:
    https://huggingface.co/AutonLab/MOMENT-1-large

    pretrained_model_name_or_path : str or None
        Path to the pretrained Momentfm model. If ``None``, initialize a model
        from ``config``. Default is AutonLab/MOMENT-1-large.

    head_dropout : float
        Dropout value of classification head of the model. Values range between
        [0.0, 1.0]
        Default = 0.1

    batch_size : int
        size of batches to train the model on
        default = 32

    eval_batch_size : int or "all"
        size of batches to evaluate the model on. If the string "all" is
        specified, then we process the entire validation set as a single batch
        default = 32

    epochs : int
        Number of epochs to fit tune the model on
        default = 1

    max_lr : float
        Maximum learning rate that the learning rate scheduler will use
        default = 1e-4

    device : str
        torch device to use
        default = "auto"
        If set to auto, it will automatically use whatever device that
        `accelerate` detects.

    pct_start : float
        percentage of total iterations where the learning rate rises during
        one epoch
        default = 0.3

    max_norm : float
        Float value used to clip gradients during training
        default = 5.0

    train_val_split : float
        float value between 0 and 1 to determine portions of training
        and validation splits
        default = 0.2

    config : dict, default = {}
        If desired, user can pass in a config detailing all momentfm parameters
        that they wish to set in dictionary form, so that parameters do not need
        to be individually set. If a parameter inside a config is a
        duplicate of one already passed in individually, it will be overwritten.
        Required if ``pretrained_model_name_or_path`` is ``None``.
        For randomly initialized T5 backbones, ``transformer_config`` may be supplied
        as a dictionary of ``transformers.T5Config`` parameters.

    to_cpu_after_fit : bool, default = False
        After fitting and training, will return the `momentfm` model to the cpu.

    References
    ----------
    Paper: https://arxiv.org/abs/2402.03885
    Github: https://github.com/moment-timeseries-foundation-model/moment/tree/main

    Examples
    --------
    >>> from sktime.classification.foundation_models.momentfm import (
    ...     MomentFMClassifier,
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_type = "numpy3d")
    >>> X_test, _ = load_unit_test(split="test", return_type = "numpy3d")
    >>> classifier = MomentFMClassifier(epochs=1, batch_size=16)
    >>> classifier.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_pred = classifier.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "python_dependencies": [
            "torch",
            "tqdm",
            "huggingface-hub",
            "hf-xet",
            # "momentfm",
            "accelerate",
            "transformers",
        ],
        "python_version": ">= 3.10",
        "serialization:native_artifacts": ("model",),
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.momentfm"],
        "tests:skip_by_name": [
            # see 8253
            "test_fit_idempotent",
            "test_multiprocessing_idempotent",
        ],
    }

    def __init__(
        self,
        pretrained_model_name_or_path="AutonLab/MOMENT-1-large",
        head_dropout=0.1,
        batch_size=32,
        eval_batch_size=32,
        epochs=1,
        max_lr=1e-4,
        device="auto",
        pct_start=0.3,
        max_norm=5.0,
        train_val_split=0.2,
        config=None,
        to_cpu_after_fit=False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.head_dropout = head_dropout
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.pct_start = pct_start
        self.max_norm = max_norm
        self.train_val_split = train_val_split
        self.config = config
        self._config = config if config is not None else {}
        self.to_cpu_after_fit = to_cpu_after_fit
        super().__init__()

    def _get_native_artifact_load_kwargs(self, name):
        """Return kwargs needed to reload native artifacts from sktime bundles."""
        if name != "model":
            return {}

        return {"model_kwargs": self._get_model_kwargs()}

    def _get_model_kwargs(self):
        """Return task-specific MOMENT model kwargs from fitted state."""
        return {
            "task_name": "classification",
            "n_channels": getattr(self, "_n_channels", 1),
            "num_class": self.n_classes_,
            "dropout": self._config.get("head_dropout", self.head_dropout),
            "device": getattr(self, "_device", "cpu"),
        }

    def __getstate__(self):
        """Return state for pickling, excluding the unpicklable torch model.

        ``MOMENTPipeline`` inherits from ``PreTrainedModel``, which registers
        a local closure (``make_inputs_require_grads``) as a forward hook when
        gradient-checkpointing is enabled.  Local closures are not picklable,
        so we strip ``self.model`` from the serialised state.  It will be
        reloaded from the multiton cache on the first ``_predict`` call after
        unpickling.
        """
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        """Restore state from an unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_momentfm_cache_key(self):
        """Build a unique cache key for the current model configuration.

        The key encodes all parameters that influence which pretrained pipeline
        is loaded so that the multiton cache can distinguish between different
        configurations.

        Returns
        -------
        str
            Deterministic cache key string.
        """
        # n_channels is determined by X in _fit and stored as self._n_channels
        return _momentfm_cache_key(
            pretrained_model_name_or_path=self._pretrained_model_name_or_path,
            n_channels=self._n_channels,
            num_class=self.n_classes_,
            dropout=self._config.get("head_dropout", self.head_dropout),
            device=self._device,
        )

    def _load_model(self):
        """Lazily load (or retrieve cached) MOMENTPipeline.

        Uses the multiton pattern via :class:`_CachedMomentFMPipeline` so that
        the expensive ``MOMENTPipeline.from_pretrained()`` call is executed at
        most once per unique combination of model path, number of channels,
        number of classes, dropout, and device.

        Returns
        -------
        MOMENTPipeline
            Initialised pipeline ready for fine-tuning.
        """
        if self._pretrained_model_name_or_path is None:
            if self.config is None:
                raise ValueError(
                    "`config` must be provided when "
                    "`pretrained_model_name_or_path=None`."
                )
            model = MOMENTPipeline(
                self._config,
                model_kwargs=self._get_model_kwargs(),
            )
            model.init()
            return model

        key = self._get_momentfm_cache_key()
        return _CachedMomentFMPipeline(
            key=key,
            pretrained_model_name_or_path=self._pretrained_model_name_or_path,
            n_channels=self._n_channels,
            num_class=self.n_classes_,
            dropout=self._config.get("head_dropout", self.head_dropout),
            device=self._device,
        ).load()

    def _fit(self, X, y):
        """MomentFMClassifier fit method.

        Parameters
        ----------
        X : This is the set of time series data that the model will fine-tune on.
            Can either be 2D or 3D numpy array. Each time series must be of
            length 512 or less, and the number of rows designates the number of
            time series in the dataset. If the time series is longer than 512,
            then we will use the most recent 512 time steps.
                2D: (num_timeseries, seq_len)
                3D: (num_timeseries, num_channels, seq_len)

        y : 1D numpy array of shape (num_timeseries,)
            This is the set of labels/classes. If the classes are integers
            of range [1, num_classes], then they will be converted to
            [0, num_classes - 1] for the model.

        """
        self.y_dtype = y.dtype

        self._pretrained_model_name_or_path = self._config.get(
            "pretrained_model_name_or_path", self.pretrained_model_name_or_path
        )
        # device initialization
        self._device = self._config.get("device", self.device)

        # check availability of user specified device
        self._device = _check_device(self._device)
        # initialize accelerator
        accelerator = accelerate.Accelerator()
        if self._device == "auto":
            self._device = accelerator.device

        cur_epoch = 0
        max_epoch = self.epochs
        self._n_channels = X.shape[1] if len(X.shape) == 3 else 1

        # Lazy init: load (or retrieve cached) MOMENTPipeline backbone.
        # Calling _load_model() ensures the expensive from_pretrained() call
        # is executed at most once per unique configuration (multiton pattern).
        self.model = self._load_model()
        # preparing the datasets
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y, X, train_size=1 - self.train_val_split, test_size=self.train_val_split
        )

        # if we need to transform the labels, do it here
        y_train, y_test, mapping, inverse_mapping = _transform_labels(y_train, y_test)

        self.mapping = mapping
        self.inverse_mapping = inverse_mapping

        train_dataset = MomentFMClassifierPytorchDataset(
            y=y_train,
            X=X_train,
            device=self._device,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        if not X_test.shape[0] == 0:
            val_dataset = MomentFMClassifierPytorchDataset(
                y=y_test,
                X=X_test,
                device=self._device,
            )

            if self.eval_batch_size == "all":
                self._eval_batch_size = len(val_dataset)
            else:
                self._eval_batch_size = self.eval_batch_size

            val_dataloader = DataLoader(
                val_dataset, batch_size=self._eval_batch_size, shuffle=True
            )
        else:
            val_dataloader = None

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=self.max_lr)
        # Enable mixed precision training

        # Create a OneCycleLR scheduler
        total_steps = len(train_dataloader) * max_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
        )

        # Gradient clipping value
        max_norm = self.max_norm

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = (
            accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler
            )
        )

        while cur_epoch < max_epoch:
            cur_epoch = _run_epoch(
                cur_epoch,
                accelerator,
                criterion,
                optimizer,
                scheduler,
                self.model,
                max_norm,
                train_dataloader,
                val_dataloader,
            )

        if self.to_cpu_after_fit:
            self.model.to("cpu")
            empty_cache()

        return self

    def _predict(self, X):
        """Predict method to classify labels for the input data.

        Parameters
        ----------
        X : 2D or 3D numpy array
            This is the set of time series data that the model will predict on.
            Can either be 2D or 3D numpy array. Each time series must be of
            length 512 or less, and the number of rows designates the number of
            time series in the dataset. If the time series is longer than 512,
            then we will use the most recent 512 time steps.
                2D: (num_timeseries, seq_len)
                3D: (num_timeseries, num_channels, seq_len)

        Returns
        -------
        y_pred : 1D numpy array
            numpy array of shape (num_timeseries,) correponding to the
            predicted classes.

        """
        from torch import from_numpy

        X_ = deepcopy(X)

        # if length of time series is greater than 512, then we will
        # use the most recent 512 time steps
        if X_.shape[-1] > 512:
            if len(X_.shape) == 2:
                X_ = X_[:, -512:]
            elif len(X.shape) == 3:
                X_ = X_[:, :, -512:]
            input_mask = np.ones(X_.shape[-1])

        # if length of time series is less than 512, then we will pad
        if X_.shape[-1] < 512:
            pad_length = 512 - X_.shape[-1]
            pad_shape = list(X_.shape)
            pad_shape[-1] = pad_length
            pad = np.zeros(pad_shape)
            X_ = np.concatenate((X_, pad), axis=-1)
            input_mask = np.concatenate((np.ones(X.shape[-1]), np.zeros(pad_length)))
        else:
            input_mask = np.ones(X_.shape[-1])

        # Reload from multiton cache if the model was cleared by unpickling
        # (``__getstate__`` sets ``self.model = None`` to avoid pickling the
        # non-serialisable HuggingFace forward hook).
        if getattr(self, "model", None) is None:
            self.model = self._load_model()

        self.model.eval()
        self.model.to(self._device)

        # Move the data to the specified device
        X_ = from_numpy(X_).to(self._device).float()

        input_mask = from_numpy(input_mask).to(self._device)

        outputs = self.model(x_enc=X_, mask=input_mask)
        logits = outputs.logits

        # Get the predicted class
        _, indices = logits.max(1)

        if self.inverse_mapping is not None:
            indices = indices.cpu().numpy()
            y_pred = np.vectorize(self.inverse_mapping.get)(indices)
        else:
            y_pred = indices.cpu().numpy()

        if self.to_cpu_after_fit:
            self.model.to("cpu")
            empty_cache()

        # convert the dtype of y_pred to the same as y
        y_pred = y_pred.astype(self.y_dtype)

        return y_pred

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_config = {
            "seq_len": 512,
            "patch_len": 8,
            "patch_stride_len": 8,
            "d_model": 8,
            "transformer_backbone": "tiny-random-t5",
            "transformer_type": "encoder_only",
            "randomly_initialize_backbone": True,
            "enable_gradient_checkpointing": False,
            "dropout": 0.1,
            "transformer_config": {
                "d_model": 8,
                "d_ff": 16,
                "num_layers": 1,
                "num_decoder_layers": 1,
                "num_heads": 2,
                "vocab_size": 32,
            },
        }
        params1 = {
            "pretrained_model_name_or_path": None,
            "config": deepcopy(test_config),
            "device": "cpu",
            "to_cpu_after_fit": True,
            "train_val_split": 0.0,
            "batch_size": 64,
        }
        params2 = {
            "pretrained_model_name_or_path": None,
            "config": deepcopy(test_config),
            "device": "cpu",
            "batch_size": 128,
            "to_cpu_after_fit": True,
            "train_val_split": 0.0,
        }

        return [params1, params2]


def _run_epoch(
    cur_epoch,
    accelerator,
    criterion,
    optimizer,
    scheduler,
    model,
    max_norm,
    train_dataloader,
    val_dataloader,
):
    import torch.cuda.amp
    from tqdm import tqdm

    losses = []
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        # Move the data to the GPU
        timeseries = data["historical_X"]
        input_mask = data["input_mask"]
        labels = data["labels"]

        with torch.cuda.amp.autocast():
            output = model(x_enc=timeseries, mask=input_mask)
        loss = criterion(output.logits, labels)

        accelerator.backward(loss)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    tqdm.write(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    # Step the learning rate scheduler
    scheduler.step()

    # Evaluate the model on the test split
    if val_dataloader:
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for data in tqdm(val_dataloader, total=len(val_dataloader)):
                # Move the data to the specified device
                timeseries = data["historical_X"]
                input_mask = data["input_mask"]
                labels = data["labels"]
                with torch.cuda.amp.autocast():
                    output = model(x_enc=timeseries, mask=input_mask)
                loss = criterion(output.logits, labels)
                running_loss += loss.item()
        avg_loss = running_loss / len(val_dataloader)
        tqdm.write(f"Epoch {cur_epoch}: Loss {avg_loss:.3f}")
    cur_epoch += 1
    return cur_epoch


def _check_device(device):
    if device == "auto":
        return device
    mps = False
    cuda = False
    if device == "mps":
        from torch.backends.mps import is_available, is_built

        if not is_available():
            if not is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            _device = "mps"
            mps = True
    elif device == "gpu" or device == "cuda":
        from torch.cuda import is_available

        if is_available():
            _device = "cuda"
            cuda = True
    elif "cuda" in device:  # for specific cuda devices like cuda:0 etc
        from torch.cuda import is_available

        if is_available():
            _device = device
            cuda = True
    if mps or cuda:
        return _device
    else:
        _device = "cpu"

    return _device


def _transform_labels(y, y_test=None):
    """Transform labels to be in the range [0, num_classes - 1].

    If the labels are already in this range, return them as is.

    Returns
    -------
    y : np.ndarray
        Transformed labels.
    y_test : np.ndarray
        Transformed test labels.
    mapping : dict
        Mapping from original labels to transformed labels.
    inverse_mapping : dict
        Mapping from transformed labels to original labels.
    """
    labels = np.unique(y)

    if np.array(list(range(len(labels))) == labels).all():
        return y, y_test, None, None

    mapping = {l: i for i, l in enumerate(labels)}

    inverse_mapping = {i: l for i, l in enumerate(labels)}
    y = np.vectorize(mapping.get)(y)

    if y_test is not None and len(y_test) > 0:
        y_test = np.vectorize(mapping.get)(y_test)

    return y, y_test, mapping, inverse_mapping


class MomentFMClassifierPytorchDataset(Dataset):
    """Customized Pytorch dataset for the momentfm model."""

    def __init__(self, y, X, device):
        from torch import from_numpy

        self.y = from_numpy(y) if isinstance(y, np.ndarray) else y
        self.X = from_numpy(X) if isinstance(X, np.ndarray) else X
        self.seq_len = 512
        self.shape = y.shape
        self.device = device

        if X.shape[-1] > self.seq_len:
            if len(self.shape) == 2:
                self.X = X[:, -self.seq_len :]
            elif len(self.shape) == 3:
                self.X = X[:, :, -self.seq_len :]
        elif X.shape[-1] < self.seq_len:
            self.pad_length = self.seq_len - X.shape[-1]

        if len(self.y.shape) == 2:
            # in case its (n, 1) change it to (n,)
            self.y = self.y.squeeze()

    def __len__(self):
        """Return length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return dataset items from index i."""
        from torch import cat, ones, zeros
        # batches must be returned in format (B, C, S)
        # where B = batch_size, C = channels, S = sequence_length

        historical_X = self.X[i, :, :].to(self.device)
        labels = self.y[i].long().to(self.device)

        if self.pad_length > 0:
            self.input_mask = cat((ones(self.X.shape[-1]), zeros(self.pad_length)))
            pad_shape = list(historical_X.shape)
            pad_shape[-1] = self.pad_length
            pad = zeros(pad_shape).to(self.device)
            historical_X = cat((historical_X, pad), dim=-1)
        else:
            self.input_mask = ones(self.seq_len).to(self.device)

        return {
            "labels": labels,
            "historical_X": historical_X.float(),
            "input_mask": self.input_mask.float(),
        }
