"""Abstract base class for the Pytorch neural network classifiers."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepClassifierPytorch"]

import abc
import tempfile
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import LabelEncoder

from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

ReduceLROnPlateau = _safe_import("torch.optim.lr_scheduler.ReduceLROnPlateau")

# Lightning callback names accepted as strings in the ``callbacks`` parameter.
# See https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
_LIGHTNING_CALLBACKS = {
    "earlystopping": "EarlyStopping",
    "modelcheckpoint": "ModelCheckpoint",
    "learningratemonitor": "LearningRateMonitor",
    "richprogressbar": "RichProgressBar",
    "tqdmprogressbar": "TQDMProgressBar",
    "devicestatsmonitor": "DeviceStatsMonitor",
}

LC_TO_UC_ACTIVATIONS = {
    "elu": "ELU",
    "hardshrink": "Hardshrink",
    "hardsigmoid": "Hardsigmoid",
    "hardtanh": "Hardtanh",
    "hardswish": "Hardswish",
    "leakyrelu": "LeakyReLU",
    "logsigmoid": "LogSigmoid",
    "multiheadattention": "MultiheadAttention",
    "prelu": "PReLU",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "rrelu": "RReLU",
    "selu": "SELU",
    "celu": "CELU",
    "gelu": "GELU",
    "sigmoid": "Sigmoid",
    "silu": "SiLU",
    "mish": "Mish",
    "softplus": "Softplus",
    "softshrink": "Softshrink",
    "softsign": "Softsign",
    "tanh": "Tanh",
    "tanhshrink": "Tanhshrink",
    "threshold": "Threshold",
    "glu": "GLU",
    "softmin": "Softmin",
    "softmax": "Softmax",
    "softmax2d": "Softmax2d",
    "logsoftmax": "LogSoftmax",
    "adaptivelogsoftmaxwithloss": "AdaptiveLogSoftmaxWithLoss",
}


class BaseDeepClassifierPytorch(BaseClassifier):
    """Abstract base class for the Pytorch neural network classifiers.

    Parameters
    ----------
    num_epochs : int, default = 100
        The number of epochs to train the model
    batch_size : int, default = 8
        The size of each mini-batch during training
    activation : str, Callable, or None, default=None
        Activation applied to the output layer.

        Permitted values:

        - ``None``: no activation is applied to the output layer and the network
          returns raw outputs (logits). This is typically required when using
          ``CrossEntropyLoss``, which expects logits as input.
        - ``str``: name of a class in ``torch.nn``. Case-sensitive names are
          recommended and must match PyTorch (e.g., ``"ReLU"``, ``"LeakyReLU"``).
          Lowercase aliases for common activations are also accepted
          (e.g., ``"relu"`` is resolved to ``"ReLU"``). The class is instantiated
          with default constructor arguments. Must be a valid ``torch.nn``
          activation; see
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - ``torch.nn.Module``: an instance of a ``torch.nn.Module`` subclass,
          for example ``torch.nn.ReLU()``. Arbitrary callables are not supported.

    criterion : case insensitive str or an instance of a loss function
        defined in PyTorch, default = None
        The loss function to be used in training the neural network.
        If None, CrossEntropyLoss is used.
        If a string/Callable is passed, it must be one of the loss functions defined in
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict or None, default = None
        The keyword arguments to be passed to the loss function.
    optimizer : case insensitive str or an instance of an optimizer
        defined in PyTorch, default = None
        The optimizer to use for training the model. If None, Adam optimizer is used.
        If a string/Callable is passed, it must be one of the optimizers defined in
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        The keyword arguments to be passed to the optimizer.
    callbacks : None, str, tuple of str and/or lightning.pytorch.callbacks.Callback,
        default = None
        Callbacks to use during training. Supports:

        * PyTorch learning rate schedulers, passed as case-insensitive strings.
          If more than one scheduler is passed, they are applied sequentially in the
          order they are passed. Since schedulers must be initialized with the
          optimizer, only class names (str) are accepted, not instances.
          List of available schedulers:
          https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        * Lightning callbacks, passed as instances of
          ``lightning.pytorch.callbacks.Callback``. Case-insensitive string names are
          also accepted when Lightning can construct the callback without arguments;
          otherwise pass a configured instance.
          See https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
        * Learning rate schedulers can also be passed as a callable that accepts the
          optimizer and returns a scheduler instance, for custom configuration.

        When any Lightning callback is specified, training is delegated to
        ``lightning.pytorch.Trainer``. Otherwise, a plain PyTorch training loop is used.

        If None, no callbacks are used.
    callback_kwargs : dict or None, default = None
        Deprecated and ignored. Pass callback or scheduler instances/callables instead.
    metrics : None or str or Callable or tuple of str and/or Callable, default = None
        Metrics to compute during training. If None, no metrics are computed beyond
        the loss. Metrics are computed from torchmetrics library.
        If a string/Callable is passed, it must be one of the metrics defined in
        https://lightning.ai/docs/torchmetrics/stable/
        Examples: "Accuracy", "F1Score", "Precision", "Recall"
    lr : float, default = 0.001
        The learning rate to be used in the optimizer.
    verbose : bool, default = True
        Whether to output extra information.
    random_state : int or None, default = None
        Seed to ensure reproducibility.
    """

    _tags = {
        "authors": ["geetu040", "RecreationalMath"],
        "maintainers": ["geetu040", "RecreationalMath"],
        "python_dependencies": ["torch"],
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:multioutput": False,
        "capability:random_state": True,
        "property:randomness": "stochastic",
        "tests:vm": True,
    }

    # _instantiate_activation_vars is an iterable of attribute names of activations
    # to instantiate. In case activation attributes in subclasses are different than
    # the default ones (activation and activation_hidden), this variable should
    # be overridden.
    _instantiate_activation_vars = ("activation", "activation_hidden")

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.metrics is not None:
            self.set_tags(**{"tests:python_dependencies": "torchmetrics"})
        if self._uses_lightning_training():
            # TODO: i hope this doesnt override the torchmetrics tag, will check
            self.set_tags(**{"tests:python_dependencies": "lightning"})

    def __init__(
        self: "BaseDeepClassifierPytorch",
        num_epochs: int = 16,
        batch_size: int = 8,
        activation: str | None | Callable = None,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict | None = None,
        optimizer: str | Callable | None = None,
        optimizer_kwargs: dict | None = None,
        callbacks: None | str | tuple[str | object, ...] = None,
        callback_kwargs: dict | None = None,
        metrics: None | str | Callable | tuple[str | Callable, ...] = None,
        lr: float = 0.001,
        verbose: bool = True,
        random_state: int | None = None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.activation = activation
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        if callback_kwargs is not None:
            warnings.warn(
                "callback_kwargs is deprecated and ignored. Pass callback or "
                "scheduler instances/callables for custom configuration.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.callback_kwargs = callback_kwargs
        self.metrics = metrics
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        # set random seed for torch
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

        # validate activation function w.r.t. criterion specified
        self._validate_activation_criterion()

        # post this function call,
        # self._validated_criterion and self._validated_activation are used
        # and self.criterion and self.activation are ignored
        activation_map = {}
        for var in self._instantiate_activation_vars:
            activation_map[var] = getattr(self, var, None)
            if var == "activation":
                activation_map[var] = self._validated_activation
        self._callable_activations = self._instantiate_activations(activation_map)
        # optimizers, criterions, callbacks will be instantiated in
        # _instantiate_optimizer, _instantiate_criterion & _instantiate_callbacks
        # methods respectively
        self._all_optimizers = None
        self._all_criterions = None
        self._all_callbacks = None
        if self.callbacks is None:
            self._callbacks = None
        elif not isinstance(self.callbacks, tuple):
            self._callbacks = (self.callbacks,)
        else:
            self._callbacks = self.callbacks

        # use this when y has str
        self.label_encoder = None
        self._metrics_objects = None

    def _fit(self, X, y):
        if self.random_state is not None:
            import torch

            torch.manual_seed(self.random_state)

        y = self._encode_y(y)

        self.network = self._build_network(X, y)

        # instantiate loss function and optimizer
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        # instantiate callbacks (learning rate schedulers and/or Lightning callbacks)
        self._schedulers = self._instantiate_schedulers()
        self._lightning_callbacks = self._instantiate_lightning_callbacks()
        # ensure num_classes is set before instantiating metrics
        # as classification metrics require num_classes as an argument
        self.num_classes = len(np.unique(y))
        # instantiate metrics
        self._metrics_objects = self._instantiate_metrics(
            self.metrics, self.num_classes
        )
        # build dataloader
        dataloader = self._build_dataloader(X, y)

        if self._lightning_callbacks is not None:
            self._fit_lightning(dataloader)
        else:
            self._fit_manual(dataloader)

    def _fit_manual(self, dataloader):
        """Train the network using a plain PyTorch loop."""
        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _fit_lightning(self, dataloader):
        """Train the network using ``lightning.pytorch.Trainer``."""
        _check_soft_dependencies("lightning", severity="error")

        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint

        lightning_module = _SktimeClassifierLightningModule(
            network=self.network,
            criterion=self._criterion,
            optimizer=self._optimizer,
            schedulers=self._schedulers,
            metrics_objects=self._metrics_objects,
        )

        enable_checkpointing = any(
            isinstance(cb, ModelCheckpoint) for cb in self._lightning_callbacks
        )
        needs_logger = any(
            cb.__class__.__name__ == "LearningRateMonitor"
            for cb in self._lightning_callbacks
        )

        trainer_kwargs = {
            "max_epochs": self.num_epochs,
            "enable_progress_bar": self.verbose,
            "enable_model_summary": self.verbose,
            "enable_checkpointing": enable_checkpointing,
            "logger": False,
        }
        if needs_logger:
            CSVLogger = _safe_import(
                "lightning.pytorch.loggers.CSVLogger", pkg_name="lightning"
            )
            trainer_kwargs["logger"] = CSVLogger(
                save_dir=tempfile.mkdtemp(prefix="sktime_lightning_logs_")
            )

        trainer = pl.Trainer(
            callbacks=self._lightning_callbacks,
            **trainer_kwargs,
        )
        trainer.fit(lightning_module, train_dataloaders=dataloader)

        # restore best checkpoint weights when ModelCheckpoint was used
        if trainer.checkpoint_callback is not None:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path:
                checkpoint = _safe_import("torch.load")(best_path, weights_only=False)
                state_dict = checkpoint["state_dict"]
                network_state = {
                    key.removeprefix("network."): value
                    for key, value in state_dict.items()
                    if key.startswith("network.")
                }
                self.network.load_state_dict(network_state)

    def _run_epoch(self, epoch, dataloader):
        losses = []
        metric_values = {name: [] for name in (self._metrics_objects or {})}

        for inputs, outputs in dataloader:
            y_pred = self.network(**inputs)
            loss = self._criterion(y_pred, outputs)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

            # Compute metrics if any
            if self._metrics_objects:
                import torch

                with torch.no_grad():
                    for metric_name, metric_obj in self._metrics_objects.items():
                        metric_value = metric_obj(y_pred, outputs)
                        metric_values[metric_name].append(metric_value.item())

        epoch_loss = np.average(losses)
        # step the schedulers, if any
        if self._schedulers:
            for scheduler in self._schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # if ReduceLROnPlateau is used,
                    # a metric value need to be passed here.
                    # We pass the loss value of the last epoch.
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        # print loss and metrics(if any) for the epoch, if verbose is True
        if self.verbose:
            msg = f"Epoch {epoch + 1}: Loss: {epoch_loss}"
            if metric_values:
                for metric_name, values in metric_values.items():
                    avg_metric = np.average(values)
                    msg += f", {metric_name}: {avg_metric:.4f}"
            print(msg)

    def _instantiate_activations(
        self, activations: dict[str, str | Callable | None]
    ) -> dict[str, Callable | None]:
        """Instantiate PyTorch activations from string or module specifications.

        Parameters
        ----------
        activations : dict[str, str | Callable | None]
            A mapping where each key is the name of an activation attribute, and the
            value is either the activation specified by the user or a default provided
            by the estimator.

        Returns
        -------
        callable_activations : dict[str, torch.nn.Module | None]
            A dictionary of activation functions, keyed by the attribute name.
        """
        import torch

        callable_activations: dict[str, torch.nn.Module | None] = {}
        for activation_var, activation in activations.items():
            if activation is None:
                callable_activations[activation_var] = None
                continue
            if isinstance(activation, torch.nn.Module):
                callable_activations[activation_var] = activation
                continue
            elif not isinstance(activation, str):
                raise TypeError(
                    f"Activation '{activation}' should be string or a torch.nn.Module. "
                    f"But got {type(activation)} instead."
                )

            uc_activation = LC_TO_UC_ACTIVATIONS.get(activation, activation)
            if not _safe_import(f"torch.nn.{uc_activation}"):
                raise ValueError(
                    f"Activation '{uc_activation}' is not a valid PyTorch activation"
                    "function in torch.nn module. Please pass a valid PyTorch"
                    "activation function in torch.nn module. Refer "
                    "https://pytorch.org/docs/stable/nn.html#non-linear-activations-"
                    "weighted-sum-nonlinearity for list of valid activation functions."
                )

            callable_activations[activation_var] = _safe_import(
                f"torch.nn.{uc_activation}"
            )()
        return callable_activations

    def _validate_activation_criterion(self):
        """Validate activation function in the output layer w.r.t. criterion specified.

        Certain PyTorch criterions expect the output layer to have no activation
        function. Such as, CrossEntropyLoss, BCEWithLogitsLoss, etc.
        While certain combinations of criterion and activation function
        are functionally equivalent to using CrossEntropyLoss with no activation,
        but using CrossEntropyLoss is preferred due to numerical stability.

        This method checks for both these cases, and either raises an error or
        chooses CrossEntropyLoss with no activation if a functionally equivalent
        combination is detected.

        Examples of such functionally equivalent combinations:
        for binary classification:
        - CrossEntropyLoss with no activation with 2 neurons in output layer
        - BCEWithLogitsLoss with no activation with 1 neuron in output layer
        - BCELoss with sigmoid activation with 1 neuron in output layer
        - NLLLoss with logsoftmax activation with 2 neurons in output layer

        for multi-class classification:
        - CrossEntropyLoss with no activation with N neurons in output layer
        - NLLLoss with logsoftmax activation with N neurons in output layer

        Sets
        ------
        self._validated_criterion : str or Callable
            The validated criterion to be used in training the neural network.
            This will either be the same as self.criterion, or "crossentropyloss"
            if a functionally equivalent combination of criterion and activation
            function is detected.
        self._validated_activation : str or Callable or None
            The validated activation function to be used in the output layer.
            This will either be the same as self.activation, or None if a
            functionally equivalent combination is detected.

        Raises
        ------
        ValueError
            If the activation function is incompatible with the chosen loss function.
        """
        if not self.criterion:
            # if no criterion is passed, use CrossEntropyLoss as default
            # and no activation in the output layer
            if self.activation is not None:
                raise ValueError(
                    "When no criterion is passed, CrossEntropyLoss is used as the "
                    "default loss function. In this case, the activation function "
                    "in the output layer must be None. "
                    f"But got activation = {self.activation}. "
                    "This is because CrossEntropyLoss in PyTorch combines LogSoftmax "
                    "and NLLLoss in one single class. Therefore, no need to apply "
                    "activation in the output layer."
                    "Refer https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"
                )
            self._validated_criterion = "crossentropyloss"
            self._validated_activation = None
            return

        # import the base class for all loss functions in PyTorch
        torchLossFunction = _safe_import("torch.nn.modules.loss._Loss")

        if isinstance(self.criterion, str):
            criterion_passed = self.criterion.lower()
        elif isinstance(self.criterion, torchLossFunction):
            # import the specific loss functions to check for functionally equivalent
            # combinations of criterion and activation function
            CrossEntropyLoss = _safe_import("torch.nn.CrossEntropyLoss")
            BCEWithLogitsLoss = _safe_import("torch.nn.BCEWithLogitsLoss")
            BCELoss = _safe_import("torch.nn.BCELoss")
            NLLLoss = _safe_import("torch.nn.NLLLoss")
            if isinstance(self.criterion, CrossEntropyLoss):
                criterion_passed = "crossentropyloss"
            elif isinstance(self.criterion, BCEWithLogitsLoss):
                criterion_passed = "bcewithlogitsloss"
            elif isinstance(self.criterion, BCELoss):
                criterion_passed = "bceloss"
            elif isinstance(self.criterion, NLLLoss):
                criterion_passed = "nllloss"
            else:
                criterion_passed = "other"
        else:
            # if criterion is neither None, nor a string nor an instance of
            # a valid PyTorch loss function, raise an error
            raise TypeError(
                "`criterion` can either be None, a str or an instance of "
                "PyTorch loss functions defined in "
                "https://pytorch.org/docs/stable/nn.html#loss-functions "
                f"But got {type(self.criterion)} instead."
            )

        # import the base class for all activation functions in PyTorch
        NNModule = _safe_import("torch.nn.modules.module.Module")

        if self.activation is None:
            activation_passed = None
        elif isinstance(self.activation, str):
            activation_passed = self.activation.lower()
        elif isinstance(self.activation, NNModule):
            # import the specific activation functions to check for
            # functionally equivalent combinations of criterion and activation function
            Sigmoid = _safe_import("torch.nn.Sigmoid")
            Softmax = _safe_import("torch.nn.Softmax")
            LogSoftmax = _safe_import("torch.nn.LogSoftmax")
            if isinstance(self.activation, Sigmoid):
                activation_passed = "sigmoid"
            elif isinstance(self.activation, Softmax):
                activation_passed = "softmax"
            elif isinstance(self.activation, LogSoftmax):
                activation_passed = "logsoftmax"
            else:
                activation_passed = "other"
        else:
            # if activation is neither None, nor a string, nor an instance of
            # a valid PyTorch activation function, raise an error
            raise TypeError(
                "`activation` can either be None, a str or an instance of a valid "
                "PyTorch activation function."
                f"But got {type(self.activation)} instead."
            )

        # now check for incompatible combinations of criterion and activation function
        # and also check for functionally equivalent combinations
        # of criterion and activation function
        # that are equivalent to using CrossEntropyLoss with no activation
        if criterion_passed == "crossentropyloss" and activation_passed is not None:
            raise ValueError(
                f"When using {self.criterion} as the loss function, "
                "the activation function in the output layer must be None. "
                f"But got activation = {self.activation}. "
                "This is because CrossEntropyLoss in PyTorch combines LogSoftmax "
                "and NLLLoss in one single class. Therefore, no need to apply "
                "activation in the output layer."
                "Refer https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"
            )
        elif criterion_passed == "bcewithlogitsloss" and activation_passed is not None:
            raise ValueError(
                f"When using {self.criterion} as the loss function, "
                "the activation function in the output layer must be None. "
                f"But got activation = {self.activation}. "
                "This is because BCEWithLogitsLoss in PyTorch combines a Sigmoid layer "
                "and the BCELoss in one single class. Therefore, no need to apply "
                "activation in the output layer."
                "Refer https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html"
            )
        elif (
            (criterion_passed == "bceloss" and activation_passed == "sigmoid")
            or (criterion_passed == "nllloss" and activation_passed == "logsoftmax")
            or (criterion_passed == "bcewithlogitsloss" and activation_passed is None)
        ):
            # all of these are functionally equivalent to using
            # nn.CrossEntropyLoss with no activation,
            # and using nn.CrossEntropyLoss is the preferred way
            # because of numerical stability
            self._validated_criterion = "crossentropyloss"
            self._validated_activation = None
        else:
            self._validated_criterion = self.criterion
            self._validated_activation = self.activation

    def _uses_lightning_training(self):
        """Whether training should use ``lightning.pytorch.Trainer``."""
        if self._callbacks is None:
            return False
        for callback in self._callbacks:
            if not isinstance(callback, str):
                return True
            if callback.lower() in _LIGHTNING_CALLBACKS:
                return True
        return False

    def _instantiate_lightning_callbacks(self):
        """Instantiate Lightning callbacks from string names or instances.

        Returns
        -------
        list or None
            Instantiated Lightning callbacks, or None if not using Lightning training.
        """
        if not self._uses_lightning_training():
            return None

        _check_soft_dependencies("lightning", severity="error")

        Callback = _safe_import(
            "lightning.pytorch.callbacks.Callback", pkg_name="lightning"
        )
        callbacks_module = _safe_import(
            "lightning.pytorch.callbacks", pkg_name="lightning"
        )

        lightning_callbacks = []
        for callback in self._callbacks:
            if isinstance(callback, str):
                if callback.lower() in _LIGHTNING_CALLBACKS:
                    callback_class_name = _LIGHTNING_CALLBACKS[callback.lower()]
                    callback_class = getattr(callbacks_module, callback_class_name)
                    lightning_callbacks.append(callback_class())
                elif callback.lower() in (self._all_callbacks or {}):
                    # LR scheduler strings are handled in _instantiate_schedulers
                    continue
                else:
                    raise ValueError(
                        f"Unknown callback: {callback}. Pass a PyTorch learning rate "
                        f"scheduler name, a supported Lightning callback name "
                        f"({', '.join(_LIGHTNING_CALLBACKS.values())}), or a "
                        "``lightning.pytorch.callbacks.Callback`` instance."
                    )
            elif isinstance(callback, Callback):
                lightning_callbacks.append(callback)
            else:
                raise TypeError(
                    "Lightning callbacks must be passed as a supported string name or "
                    f"an instance of lightning.pytorch.callbacks.Callback. "
                    f"But got {type(callback)} instead."
                )
        return lightning_callbacks

    def _instantiate_schedulers(self):
        """Instantiate PyTorch learning rate schedulers for training.

        Schedulers are stepped manually in the plain PyTorch loop, or from
        ``_SktimeClassifierLightningModule.on_train_epoch_end`` when using Lightning.

        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.

        Returns
        -------
        list or None
            The list of instantiated schedulers to be used during training.
        """
        if self._callbacks is None:
            return None

        if self._all_callbacks is None:
            self._all_callbacks = {
                "lambdalr": "LambdaLR",
                "multiplicativelr": "MultiplicativeLR",
                "steplr": "StepLR",
                "multisteplr": "MultiStepLR",
                "constantlr": "ConstantLR",
                "linearlr": "LinearLR",
                "exponentiallr": "ExponentialLR",
                "polynomiallr": "PolynomialLR",
                "cosineannealinglr": "CosineAnnealingLR",
                "chainedscheduler": "ChainedScheduler",
                "sequentiallr": "SequentialLR",
                "reducelronplateau": "ReduceLROnPlateau",
                "cycliclr": "CyclicLR",
                "onecyclelr": "OneCycleLR",
                "cosineannealingwarmrestarts": "CosineAnnealingWarmRestarts",
            }

        schedulers = []
        Callback = _safe_import(
            "lightning.pytorch.callbacks.Callback", pkg_name="lightning"
        )
        for scheduler in self._callbacks:
            if isinstance(scheduler, str):
                if scheduler.lower() not in self._all_callbacks:
                    if self._uses_lightning_training():
                        continue
                    raise ValueError(
                        f"Unknown learning rate scheduler: {scheduler}. "
                        f"Please pass one/many of {', '.join(self._all_callbacks)} "
                        "as a callback."
                    )
                scheduler_class = _safe_import(
                    f"torch.optim.lr_scheduler.{self._all_callbacks[scheduler.lower()]}"
                )
                schedulers.append(scheduler_class(self._optimizer))
            elif callable(scheduler) and not (
                Callback is not None and isinstance(scheduler, Callback)
            ):
                schedulers.append(scheduler(self._optimizer))
        return schedulers or None

    def _instantiate_optimizer(self):
        # if no optimizer is passed, use Adam as default
        if not self.optimizer:
            opt = _safe_import("torch.optim.Adam")(
                self.network.parameters(), lr=self.lr
            )
            return opt
        if self._all_optimizers is None:
            self._all_optimizers = {
                "adadelta": "Adadelta",
                "adagrad": "Adagrad",
                "adam": "Adam",
                "adamw": "AdamW",
                "sparseadam": "SparseAdam",
                "adamax": "Adamax",
                "asgd": "ASGD",
                "lbfgs": "LBFGS",
                "nadam": "NAdam",
                "radam": "RAdam",
                "rmsprop": "RMSprop",
                "rprop": "Rprop",
                "sgd": "SGD",
            }
        # import the base class for all optimizers in PyTorch
        torchOptimizer = _safe_import("torch.optim.Optimizer")
        # if optimizer is a string, look it up in the available optimizers
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() in self._all_optimizers:
                optimizer_class = _safe_import(
                    f"torch.optim.{self._all_optimizers[self.optimizer.lower()]}"
                )
                if self.optimizer_kwargs:
                    return optimizer_class(
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    return optimizer_class(self.network.parameters(), lr=self.lr)
            else:
                raise ValueError(
                    f"Unknown optimizer: {self.optimizer}. Please pass one of "
                    f"{', '.join(self._all_optimizers)} for `optimizer`."
                )
        # if optimizer is already an instance of torch.optim.Optimizer, use it directly
        elif isinstance(self.optimizer, torchOptimizer):
            return self.optimizer
        # if optimizer is neither a string nor an instance of
        # a valid PyTorch optimizer, raise an error
        else:
            raise TypeError(
                "`optimizer` can either be None, a str or an instance of "
                "optimizers defined in torch.optim. "
                "See https://pytorch.org/docs/stable/optim.html#algorithms. "
                f"But got {type(self.optimizer)} instead."
            )

    def _instantiate_criterion(self):
        # if no criterion is passed, use CrossEntropyLoss as default
        if not self._validated_criterion:
            loss = _safe_import("torch.nn.CrossEntropyLoss")()
            return loss
        if self._all_criterions is None:
            self._all_criterions = {
                "l1loss": "L1Loss",
                "mseloss": "MSELoss",
                "crossentropyloss": "CrossEntropyLoss",
                "ctcloss": "CTCLoss",
                "nllloss": "NLLLoss",
                "poissonnllloss": "PoissonNLLLoss",
                "gaussiannllloss": "GaussianNLLLoss",
                "kldivloss": "KLDivLoss",
                "bceloss": "BCELoss",
                "bcewithlogitsloss": "BCEWithLogitsLoss",
                "marginrankingloss": "MarginRankingLoss",
                "hingeembeddingloss": "HingeEmbeddingLoss",
                "multilabelmarginloss": "MultiLabelMarginLoss",
                "huberloss": "HuberLoss",
                "smoothl1loss": "SmoothL1Loss",
                "softmarginloss": "SoftMarginLoss",
                "multilabelsoftmarginloss": "MultiLabelSoftMarginLoss",
                "cosineembeddingloss": "CosineEmbeddingLoss",
                "multimarginloss": "MultiMarginLoss",
                "tripletmarginloss": "TripletMarginLoss",
                "tripletmarginwithdistanceloss": "TripletMarginWithDistanceLoss",
            }
        # import the base class for all loss functions in PyTorch
        torchLossFunction = _safe_import("torch.nn.modules.loss._Loss")
        # if criterion is a string, look it up in the available criterions
        if isinstance(self._validated_criterion, str):
            if self._validated_criterion.lower() in self._all_criterions:
                criterion_class = _safe_import(
                    f"torch.nn.{self._all_criterions[self._validated_criterion.lower()]}"
                )
                if self.criterion_kwargs:
                    return criterion_class(**self.criterion_kwargs)
                else:
                    return criterion_class()
            else:
                raise ValueError(
                    f"Unknown criterion: {self._validated_criterion}. Please pass one "
                    f"of {', '.join(self._all_criterions)} for `criterion`."
                )
        # if criterion is already an instance of torch.nn.modules.loss._Loss, use it
        elif isinstance(self._validated_criterion, torchLossFunction):
            return self._validated_criterion
        else:
            # if criterion is neither a string nor an instance of
            # a valid PyTorch loss function, raise an error
            raise TypeError(
                "`criterion` can either be None, a str or an instance of "
                "loss functions defined in "
                "https://pytorch.org/docs/stable/nn.html#loss-functions "
                f"But got {type(self._validated_criterion)} instead."
            )

    def _instantiate_metric(self, metric, torchmetrics, num_classes):
        """Instantiate a single classification metric from torchmetrics.

        Parameters
        ----------
        metric : str or Callable
            Metric name from torchmetrics or a metric instance.
        torchmetrics : module
            The torchmetrics module.
        num_classes : int
            The number of classes in the dataset.

        Returns
        -------
        metric_name : str
            Name to use as the key in the metrics dictionary.
        metric_instance : Callable
            The instantiated metric object.

        Raises
        ------
        ValueError
            If an unknown metric name is passed.
        TypeError
            If metric is neither a string nor a callable.
        """
        if isinstance(metric, str):
            if not hasattr(torchmetrics, metric):
                raise ValueError(
                    f"Error in constructing torch based classifier "
                    f"{type(self).__name__}, "
                    f"unknown metric: {metric}. Please pass one of the available "
                    f"classification metrics from torchmetrics or check the metric "
                    f"name. See https://lightning.ai/docs/torchmetrics/stable/"
                )
            metric_class = getattr(torchmetrics, metric)
            kwargs = {"task": "multiclass", "num_classes": num_classes}
            if metric in ("F1Score", "Precision", "Recall"):
                kwargs["average"] = "macro"
            return metric, metric_class(**kwargs)
        if isinstance(metric, Callable):
            return metric.__class__.__name__, metric
        raise TypeError(
            "`metrics` can either be None, a str or a tuple of str "
            "representing metrics from torchmetrics, or an instance of a "
            f"torchmetrics metric. But got {type(metric)} instead."
        )

    def _instantiate_metrics(self, metrics, num_classes):
        """Instantiate metrics to be computed during training.

        Metrics are computed from the torchmetrics library. If no metrics are passed,
        returns None.

        Parameters
        ----------
        metrics : None or str or Callable or tuple of str and/or Callable
            Metrics to compute during training. If None, no metrics are computed beyond
            the loss. Metrics are computed from torchmetrics library.
            If a string/Callable is passed, it must be one of the metrics defined in
            https://lightning.ai/docs/torchmetrics/stable/
            Examples: "MeanSquaredError", "MeanAbsoluteError", "R2Score"
        num_classes : int
            The number of classes in the dataset.
            This is required for classification metrics.

        Returns
        -------
        metrics_dict : dict or None
            A dictionary mapping metric names to metric objects from torchmetrics.
            If no metrics are provided, returns None.

        Raises
        ------
        ValueError
            If an unknown metric name is passed.
        TypeError
            If metric is neither a string nor a callable.
        """
        if metrics is None:
            return None

        torchmetrics = _safe_import("torchmetrics")

        if not isinstance(metrics, tuple):
            metrics_list = (metrics,)
        else:
            metrics_list = metrics

        metrics_dict = {}
        for metric in metrics_list:
            metric_name, metric_instance = self._instantiate_metric(
                metric, torchmetrics, num_classes
            )
            metrics_dict[metric_name] = metric_instance

        return metrics_dict if metrics_dict else None

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        # default behaviour if estimator doesnot implement
        # dataloader of its own
        dataset = PytorchDataset(X, y)
        DataLoader = _safe_import("torch.utils.data.DataLoader")
        return DataLoader(dataset, self.batch_size)

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : should be of mtype in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted class labels
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
        y_pred_prob = self._predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        y_decoded = self._decode_y(y_pred)
        return y_decoded

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        Fsoftmax = _safe_import("torch.nn.functional.softmax")
        cat = _safe_import("torch.cat")

        self.network.eval()
        dataloader = self._build_dataloader(X)
        y_pred = []
        torchNo_grad = _safe_import("torch.no_grad")
        # disable gradient calculation for inference
        with torchNo_grad():
            for inputs in dataloader:
                y_pred.append(self.network(**inputs).detach())
        y_pred = cat(y_pred, dim=0)
        # (batch_size, num_outputs)

        # if we had self._validated_activation, then it has already been applied
        # in forward pass of the network. If not, we apply softmax here to convert
        # logits to probabilities.
        if self._validated_activation is None:
            y_pred = Fsoftmax(y_pred, dim=-1)

        y_pred = y_pred.numpy()
        return y_pred

    def _encode_y(self, y):
        unique = np.unique(y)
        if np.array_equal(unique, np.arange(len(unique))):
            return y

        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(y)

    def _decode_y(self, y):
        if self.label_encoder is None:
            return y

        return self.label_encoder.inverse_transform(y)

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch classifiers.

        PyTorch classifiers require 3D input and we don't allow automatic conversion
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch classifiers require properly "
                f"formatted 3D time series data. Please reshape your data or "
                "use a supported Panel mtype."
            )

        # Call parent method for other conversions
        return super()._internal_convert(X, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return []


Dataset = _safe_import("torch.utils.data.Dataset")


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning classifier based on pytorch."""

    def __init__(self, X, y=None):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        torchTensor = _safe_import("torch.tensor")
        torchFLoat = _safe_import("torch.float")
        torchLong = _safe_import("torch.long")
        x = self.X[i]
        x = torchTensor(x, dtype=torchFLoat)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torchTensor(y, dtype=torchLong)
        return inputs, y


LightningModule = _safe_import("lightning.LightningModule", pkg_name="lightning")


class _SktimeClassifierLightningModule(LightningModule):
    """Thin Lightning wrapper around a sktime PyTorch classification network."""

    def __init__(
        self,
        network,
        criterion,
        optimizer,
        schedulers=None,
        metrics_objects=None,
    ):
        super().__init__()
        self.network = network
        self.criterion = criterion
        self._optimizer = optimizer
        self._schedulers = schedulers or []
        self._metrics_objects = metrics_objects or {}

    def training_step(self, batch, batch_idx):
        inputs, y = batch
        y_pred = self.network(**inputs)
        loss = self.criterion(y_pred, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self._metrics_objects:
            import torch

            with torch.no_grad():
                for metric_name, metric_obj in self._metrics_objects.items():
                    metric_value = metric_obj(y_pred, y)
                    self.log(
                        f"train_{metric_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
        return loss

    def on_train_epoch_end(self):
        if not self._schedulers:
            return

        epoch_loss = self.trainer.callback_metrics.get("train_loss")
        if epoch_loss is not None:
            epoch_loss = epoch_loss.item()

        for scheduler in self._schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

    def configure_optimizers(self):
        return self._optimizer
