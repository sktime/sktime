"""Shared components for Keras-based deep learning estimators."""

from dataclasses import asdict, dataclass
from typing import Any

from sktime.base._base import BaseObject
from sktime.utils.dependencies import _check_dl_dependencies


@dataclass
class KerasCompileKwargs(BaseObject):
    """Additional keyword arguments for Keras model compilation.

    Parameters
    ----------
    loss_weights : list of float or dict, default=None
        Scalar coefficients to weight the loss contributions of different model outputs.
    weighted_metrics : list of str, default=None
        List of metrics evaluated and weighted by `sample_weight` or `class_weight`.
    run_eagerly : bool, default=False
        If True, model will run eagerly (not as a compiled graph). Useful for debugging.
    steps_per_execution : int, default=1
        Number of batches to run during each execution call.
    jit_compile : str or bool, default="auto"
        Whether to compile the model with XLA. "auto" enables for supported platforms.
    auto_scale_loss : bool, default=True
        Whether to automatically scale loss for mixed precision training.

    See Also
    --------
    Keras compile documentation :
        https://keras.io/api/models/model_training_apis/#compile-method
    """

    _tags = {
        "authors": ["achieveordie"],
        "maintainers": ["achieveordie"],
        "python_dependencies": "tensorflow",
    }

    loss_weights: list[float] | dict[str, float] | None = None
    weighted_metrics: list[str] | None = None
    run_eagerly: bool = False
    steps_per_execution: int = 1
    jit_compile: str | bool = "auto"
    auto_scale_loss: bool = True

    def __post_init__(self) -> None:
        _check_dl_dependencies(severity="error")

    def as_dict(self) -> dict[Any, Any]:
        return asdict(self)


@dataclass
class KerasFitKwargs(BaseObject):
    """Additional keyword arguments for Keras model training.

    Parameters
    ----------
    validation_split : float, default=0.0
        Fraction of training data to use as validation data (between 0.0 and 1.0).
    validation_data : tuple, tf.data.Dataset, or keras.utils.PyDataset, default=None
        Data on which to evaluate loss and metrics at the end of each epoch.
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.
    class_weight : dict, default=None
        Dictionary mapping class indices to weights for weighted loss calculation.
    sample_weight : array-like, default=None
        Weights for individual samples in the training data.
    initial_epoch : int, default=0
        Epoch at which to start training (useful for resuming training).
    steps_per_epoch : int, default=None
        Number of batches to process before considering one epoch complete.
    validation_steps : int, default=None
        Number of batches to draw from validation data before stopping.
    validation_batch_size : int, default=None
        Batch size for validation data. If None, defaults to `batch_size`.
    validation_freq : int, default=1
        Run validation every N epochs.

    See Also
    --------
    Keras fit documentation :
        https://keras.io/api/models/model_training_apis/#fit-method
    """

    _tags = {
        "authors": ["achieveordie"],
        "maintainers": ["achieveordie"],
        "python_dependencies": "tensorflow",
    }

    validation_split: float = 0.0
    validation_data: Any | None = None
    shuffle: bool = True
    class_weight: dict[int, float] | None = None
    sample_weight: Any | None = None
    initial_epoch: int = 0
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    validation_batch_size: int | None = None
    validation_freq: int = 1

    def __post_init__(self) -> None:
        _check_dl_dependencies(severity="error")

        from keras.utils import PyDataset
        from tensorflow.python.data import Dataset

        if not (0.0 <= self.validation_split <= 1.0):
            raise ValueError(
                "`validation_split` can only be between [0.0, 1.0]. "
                f"But `{self.validation_split}` was provided."
            )

        if self.validation_data is not None and not isinstance(
            self.validation_data, (tuple, Dataset, PyDataset)
        ):
            raise ValueError(
                "`validation_data` must either be a 2-length tuple, "
                "a `keras.utils.PyDataset` or `tf.data.DataSet` instance. "
                f"Found type: {type(self.validation_data)} instead."
            )

    def as_dict(self) -> dict[Any, Any]:
        return asdict(self)
