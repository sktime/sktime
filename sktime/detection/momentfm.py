"""Interface for the momentfm deep learning time series anomaly detector."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _safe_import

accelerate = _safe_import("accelerate")
Dataset = _safe_import("torch.utils.data.Dataset")
torch = _safe_import("torch")


class MomentFMAnomalyDetector(BaseDetector):
    """
    Interface for anomaly detection with the deep learning time series model momentfm.

    MomentFM is a collection of open source foundation models for the general
    purpose of time series analysis. The Moment Foundation Model is a pre-trained
    model that is capable of accomplishing various time series tasks, such as:
        - Anomaly Detection

    This interface with MomentFM focuses on the anomaly detection task, in which the
    foundation model uses its pre-trained reconstruction head to reconstruct input
    time series. Anomalies are detected by computing the reconstruction error
    (e.g., MSE or MAE) between the observed and reconstructed values.

    MOMENT supports both zero-shot anomaly detection (without fine-tuning) and
    fine-tuning for improved performance.

    For zero-shot anomaly detection, set freeze_encoder=True, freeze_embedder=True,
    and freeze_head=True, which will use the pre-trained model.
    For fine-tuning, the default and recommended approach is to train a linear probing
    head by setting freeze_encoder=True, freeze_embedder=True, and freeze_head=False,
    which will train a reconstruction head while keeping the encoder & embedder frozen.

    NOTE: This model can only handle time series with a sequence length of 512 or less.

    For more information: see
    https://github.com/moment-timeseries-foundation-model/moment

    For information regarding licensing and use of the momentfm model please visit:
    https://huggingface.co/AutonLab/MOMENT-1-large

    pretrained_model_name_or_path : str
        Path to the pretrained Momentfm model. Default is AutonLab/MOMENT-1-large

    freeze_encoder : bool, default=True
        Selection of whether or not to freeze the weights of the encoder during
        fine-tuning.

    freeze_embedder : bool, default=True
        Selection whether or not to freeze the patch embedding layer during
        fine-tuning.

    freeze_head : bool, default=False
        Selection whether or not to freeze the reconstruction head during fine-tuning.
        When freeze_encoder=True, freeze_embedder=True, and freeze_head=False,
        this enables linear probing.

    dropout : float, default=0.1
        Dropout value of the model. Values range between [0.0, 1.0]

    head_dropout : float, default=0.1
        Dropout value of the reconstruction head. Values range between [0.0, 1.0]


    batch_size : int, default=32
        Size of batches to use during inference. Also used during fine-tuning.

    eval_batch_size : int or "all", default=32
        Size of batches for evaluation. If the string "all" is specified, the
        entire validation set is processed as a single batch.

    epochs : int, default = 1
        Number of epochs to fit tune the model on.

    max_lr : float, default=1e-4
        Maximum learning rate for the OneCycleLR scheduler during fine-tuning.

    device : str, default="auto"
        Torch device to use. If "auto", will automatically use the device that
        the accelerate library detects.

    pct_start : float, default=0.3
        Percentage of total iterations where the learning rate rises during
        one epoch of fine-tuning.

    max_norm : float, default=5.0
        Maximum norm value used to clip gradients during fine-tuning.

    train_val_split : float, default=0.2
        Float value between 0 and 1 to determine portions of training and
        validation splits during fine-tuning.

    mask_ratio : float, default=0.3
        Ratio of patches to mask during fine-tuning. During pre-training, MOMENT
        learns to reconstruct randomly masked patches, so continuing this approach
        during fine-tuning improves performance.

    transformer_backbone : str, default='google/flan-t5-large'
        d_model of a pre-trained transformer model to use.

    criterion : criterion, default=torch.nn.MSELoss
        Criterion to use during fine-tuning.

    anomaly_criterion : str, default='mse'
        Metric used to compute anomaly scores. Options are 'mse' (mean squared error)
        or 'mae' (mean absolute error). The anomaly score is computed as the
        difference between observed and reconstructed time series values.

    anomaly_percentile : float, default=95.0
        Upper percentile of the anomaly score distribution used to define anomalies
        (e.g., 95 means top 5% highest scores are flagged).

    config : dict, default={}
        If desired, user can pass in a config detailing all momentfm parameters
        that they wish to set in dictionary form, so that parameters do not need
        to be individually set. If a parameter inside a config is a
        duplicate of one already passed in individually, it will be overwritten.

    return_model_to_cpu : bool, default=False
        After fitting and training, will return the momentfm model to the cpu.

    References
    ----------
    Paper: https://arxiv.org/abs/2402.03885
    Github: https://github.com/moment-timeseries-foundation-model/moment/tree/main

    Examples
    --------
    >>> from sktime.detection.momentfm import MomentFMAnomalyDetector
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample time series data
    >>> X = pd.DataFrame(np.random.randn(100, 1))
    >>> detector = MomentFMAnomalyDetector()
    >>> detector.fit(X)  # doctest: +SKIP
    >>> # Use .predict to get indices of detected anomalies
    >>> y_pred = detector.predict(X)  # doctest: +SKIP
    >>> # Use .predict_scores to get anomaly scores
    >>> y_pred = detector.predict_scores(X)  # doctest: +SKIP
    >>> # Use .transform_scores to get scores in a DataFrame format
    >>> y_pred = detector.transform_scores(X)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["mononitogoswami", "KonradSzafer", "raycai420", "Arjun7m"],
        "maintainers": ["Faakhir30"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "python_dependencies": [
            "torch",
            "tqdm",
            "huggingface-hub",
            "hf-xet",
            "accelerate",
            "transformers",
        ],
        "python_version": ">= 3.10",
        "tests:vm": True,
        "tests:libs": ["sktime.libs.momentfm"],
    }

    def __init__(
        self,
        pretrained_model_name_or_path="AutonLab/MOMENT-1-large",
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        dropout=0.1,
        head_dropout=0.1,
        batch_size=32,
        eval_batch_size=32,
        epochs=1,
        max_lr=1e-4,
        device="auto",
        pct_start=0.3,
        max_norm=5.0,
        train_val_split=0.2,
        mask_ratio=0.3,
        transformer_backbone="google/flan-t5-large",
        criterion=None,
        anomaly_criterion="mse",
        anomaly_percentile=95.0,
        config=None,
        return_model_to_cpu=False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.freeze_encoder = freeze_encoder
        self.freeze_embedder = freeze_embedder
        self.freeze_head = freeze_head
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.pct_start = pct_start
        self.max_norm = max_norm
        self.train_val_split = train_val_split
        self.mask_ratio = mask_ratio
        self.transformer_backbone = transformer_backbone
        self.criterion = criterion
        self.anomaly_criterion = anomaly_criterion
        self.anomaly_percentile = anomaly_percentile
        self.config = config
        self._config = config if config is not None else {}
        self.return_model_to_cpu = return_model_to_cpu

        self._pretrained_model_name_or_path = self._config.get(
            "pretrained_model_name_or_path", self.pretrained_model_name_or_path
        )
        self._freeze_encoder = self._config.get("freeze_encoder", self.freeze_encoder)
        self._freeze_embedder = self._config.get(
            "freeze_embedder", self.freeze_embedder
        )
        self._freeze_head = self._config.get("freeze_head", self.freeze_head)
        self._dropout = self._config.get("dropout", self.dropout)
        self._head_dropout = self._config.get("head_dropout", self.head_dropout)
        self._transformer_backbone = self._config.get(
            "transformer_backbone", self.transformer_backbone
        )
        self._device = self._config.get("device", self.device)
        self._device = _check_device(self._device)

        super().__init__()

    def _fit(self, X, y=None):
        """Fit method for MomentFMAnomalyDetector.

        For zero-shot anomaly detection, this method
        simply loads the pretrained model. For fine-tuning mode, it trains
        the reconstruction head on the data X.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data to fit the model on. Can be 2D (num_timeseries, seq_len)
            or 3D (num_timeseries, num_channels, seq_len). Each time series must be
            of length 512 or less. If longer, only the most recent 512 steps are used.

        y : None or pd.DataFrame, optional
            For supervised anomaly detection, this could contain anomaly labels.
            Currently unused for zero-shot mode. Reserved for future extensions.

        Returns
        -------
        self : reference to self
        """
        MOMENTPipeline = _safe_import("sktime.libs.momentfm.MOMENTPipeline")

        # Load the model in reconstruction mode for anomaly detection
        self.model = MOMENTPipeline.from_pretrained(
            self._pretrained_model_name_or_path,
            model_kwargs={
                "task_name": "reconstruction",
                "n_channels": X.shape[1] if len(X.shape) > 1 else 1,
                "dropout": self._dropout,
                "head_dropout": self._head_dropout,
                "freeze_encoder": self._freeze_encoder,
                "freeze_embedder": self._freeze_embedder,
                "freeze_head": self._freeze_head,
                "transformer_backbone": self._transformer_backbone,
            },
        )
        self.model.init()

        if self._freeze_encoder and self._freeze_embedder and self._freeze_head:
            # For zero-shot, just move to device
            accelerator = accelerate.Accelerator()
            if self._device == "auto":
                self._device = accelerator.device
            self.model = self.model.to(self._device)
        else:
            self._finetune(X)

        if self.return_model_to_cpu:
            self.model.to("cpu")
            empty_cache = _safe_import("torch.cuda.empty_cache")
            empty_cache()

        return self

    def _finetune(self, X):
        """Fine-tune the reconstruction head on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data to fine-tune on.
        """
        Masking = _safe_import("sktime.libs.momentfm.utils.masking.Masking")
        OneCycleLR = _safe_import("torch.optim.lr_scheduler.OneCycleLR")
        DataLoader = _safe_import("torch.utils.data.DataLoader")

        # Initialize accelerator
        accelerator = accelerate.Accelerator()
        if self._device == "auto":
            self._device = accelerator.device

        # Split data into train and validation sets
        X_train, X_val = temporal_train_test_split(
            X, train_size=1 - self.train_val_split
        )

        # Prepare dataset and dataloaders
        train_dataset = MomentFMAnomalyDetectorPytorchDataset(X=X_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create validation dataloader if validation data exists
        val_dataloader = None
        if len(X_val) > 0:
            val_dataset = MomentFMAnomalyDetectorPytorchDataset(X=X_val)
            val_batch_size = (
                len(val_dataset)
                if self.eval_batch_size == "all"
                else self.eval_batch_size
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=False
            )

        Adam = _safe_import("torch.optim.Adam")
        MSELoss = _safe_import("torch.nn.MSELoss")

        optimizer = Adam(self.model.parameters(), lr=self.max_lr)
        criterion = MSELoss() if self.criterion is None else self.criterion

        # Create OneCycleLR scheduler
        total_steps = len(train_dataloader) * self.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
        )

        max_norm = self.max_norm

        # patch masking generator
        mask_generator = Masking(mask_ratio=self.mask_ratio)

        self.model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )

        if val_dataloader is not None:
            val_dataloader = accelerator.prepare(val_dataloader)

        for cur_epoch in range(self.epochs):
            _run_epoch(
                cur_epoch,
                accelerator,
                criterion,
                optimizer,
                scheduler,
                self.model,
                max_norm,
                train_dataloader,
                mask_generator,
                val_dataloader,
            )

    def _predict(self, X):
        """Return indices of anomalies detected in the input data.

        Uses `self.anomaly_percentile` to determine the threshold for flagging
        anomalies based on the distribution of anomaly scores.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to anomaly detection.

        Returns
        -------
        y_pred : pd.Series
            Indices of detected anomalies. Empty Series if no anomalies detected.
        """
        scores = self._predict_scores(X)
        scores_np = scores.to_numpy()

        threshold = np.percentile(scores_np, self.anomaly_percentile)

        anomaly_indices = np.where(scores_np > threshold)[0]

        # Return empty sparse representation if no anomalies found
        if len(anomaly_indices) == 0:
            return self._empty_sparse()

        return pd.Series(anomaly_indices)

    def _predict_scores(self, X):
        """Return anomaly scores for all timesteps.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to anomaly detection.

        Returns
        -------
        scores : pd.Series
            Anomaly scores for each timestep in X. Higher scores indicate
            higher likelihood of anomaly.
        """
        # Convert to numpy format expected by model
        X_np = X.to_numpy()
        # Ensure 3D format: (num_timeseries, n_channels, seq_len)
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(1, 1, -1)
        elif len(X_np.shape) == 2:
            # If (num_samples, seq_len), reshape to (num_samples, 1, seq_len)
            if X_np.shape[0] > X_np.shape[1]:
                X_np = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            else:
                X_np = X_np.reshape(1, X_np.shape[0], X_np.shape[1])

        # Truncate to max length 512 if needed
        if X_np.shape[2] > 512:
            X_np = X_np[:, :, -512:]

        # Pad to length 512 if needed
        if X_np.shape[2] < 512:
            pad_length = 512 - X_np.shape[2]
            X_np = np.pad(X_np, ((0, 0), (0, 0), (pad_length, 0)), mode="constant")

        # Convert to tensor
        X_tensor = torch.from_numpy(X_np).float().to(self._device)

        # Create input mask (all ones since no missing values)
        batch_size = X_tensor.shape[0]
        input_mask = torch.ones((batch_size, 512)).to(self._device)

        # Get reconstruction from model
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_enc=X_tensor, input_mask=input_mask)
            reconstruction = output.reconstruction

        # Compute anomaly scores
        if self.anomaly_criterion == "mse":
            scores = torch.mean((X_tensor - reconstruction) ** 2, dim=(0, 1))
        elif self.anomaly_criterion == "mae":
            scores = torch.mean(torch.abs(X_tensor - reconstruction), dim=(0, 1))
        else:
            raise ValueError(
                f"Unknown anomaly_criterion: {self.anomaly_criterion}. "
                "Must be 'mse' or 'mae'."
            )

        # Convert to numpy and return only the non-padded portion
        scores_np = scores.cpu().numpy()
        scores_np = scores_np[512 - X.shape[0] :]

        return pd.Series(scores_np, index=X.index)

    def _transform_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to anomaly detection.

        Returns
        -------
        scores : pd.DataFrame
            Anomaly scores for each timestep in X, same index as X.
        """
        scores = self._predict_scores(X)
        return pd.DataFrame(scores, columns=["scores"])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        params1 = {
            "pretrained_model_name_or_path": "AutonLab/MOMENT-1-small",
            "transformer_backbone": "google/flan-t5-small",
        }
        params2 = {**params1, "anomaly_criterion": "mae"}

        return [params1, params2]


def _check_device(device):
    """Check if the specified device is available.

    Parameters
    ----------
    device : str
        Device string like "auto", "cpu", "cuda", "mps", etc.

    Returns
    -------
    str
        The device to use.
    """
    if device == "auto":
        return device

    if device == "cpu":
        return device

    if "cuda" in device.lower():
        if torch.cuda.is_available():
            return device
        else:
            raise ValueError(
                "CUDA device requested but not available. Available devices: cpu"
            )

    if "mps" in device.lower():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return device
        else:
            raise ValueError(
                "MPS device requested but not available. Available devices: cpu"
            )

    return device


class MomentFMAnomalyDetectorPytorchDataset(Dataset):
    """PyTorch Dataset for MomentFMAnomalyDetector."""

    def __init__(self, X):
        """Initialize the dataset.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input time series data.
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X

        # Ensure 2D format
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.n_channels = self.X.shape[1]

    def __len__(self):
        """Return the number of samples."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Get a single sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple of torch.Tensor
            (X_sample, input_mask) where X_sample has shape (n_channels, seq_len)
        """
        x_sample = self.X[idx : idx + 1]

        # Reshape to (seq_len, n_channels) for proper handling
        if len(x_sample.shape) == 2:
            x_sample = x_sample.T

        # Ensure length 512
        if x_sample.shape[1] > 512:
            x_sample = x_sample[:, -512:]
        elif x_sample.shape[1] < 512:
            pad_length = 512 - x_sample.shape[1]
            x_sample = np.pad(x_sample, ((0, 0), (pad_length, 0)), mode="constant")

        # Create input mask (all ones)
        input_mask = np.ones(512)

        return torch.from_numpy(x_sample).float(), torch.from_numpy(input_mask).float()


def _run_epoch(
    cur_epoch,
    accelerator,
    criterion,
    optimizer,
    scheduler,
    model,
    max_norm,
    train_dataloader,
    mask_generator,
    val_dataloader=None,
):
    """Run a single epoch of training.

    Parameters
    ----------
    cur_epoch : int
        Current epoch number.
    accelerator : accelerate.Accelerator
        Accelerator for distributed training.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    model : torch.nn.Module
        Model to train.
    max_norm : float
        Maximum gradient norm for clipping.
    train_dataloader : torch.utils.data.DataLoader
        Training data loader.
    mask_generator : Masking
        Masking object for generating patch masks during training.
    val_dataloader : torch.utils.data.DataLoader, optional
        Validation data loader.

    """
    from tqdm import tqdm

    model.train()

    losses = []
    for batch_x, batch_input_mask in tqdm(
        train_dataloader, total=len(train_dataloader)
    ):
        mask = (
            mask_generator.generate_mask(x=batch_x, input_mask=batch_input_mask)
            .to(batch_x.device)
            .long()
        )

        output = model(x_enc=batch_x, input_mask=batch_input_mask, mask=mask)

        loss = criterion(output.reconstruction, batch_x)
        losses.append(loss.item())

        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

    losses = np.array(losses)
    average_loss = np.average(losses)
    tqdm.write(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    if val_dataloader:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_input_mask in tqdm(
                val_dataloader, total=len(val_dataloader)
            ):
                output = model(x_enc=batch_x, input_mask=batch_input_mask)
                loss = criterion(output.reconstruction, batch_x)
                val_losses.append(loss.item())

        val_losses = np.array(val_losses)
        average_val_loss = np.average(val_losses)
        tqdm.write(f"Epoch {cur_epoch}: Validation loss: {average_val_loss:.3f}")
