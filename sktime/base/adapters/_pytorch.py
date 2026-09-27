"""Implements adapter for pytorch deep learning estimators."""

__all__ = ["_PytorchDeepAdapter"]
__authors__ = ["geetu040", "RecreationalMath", "srupat"]

import abc
from collections.abc import Callable

import numpy as np

from sktime.utils._lookup import _lc_class_dict, _lookup_class
from sktime.utils.dependencies import _safe_import

TORCH_NN = "torch.nn"
TORCH_MODULE = "torch.nn.Module"
TORCH_LOSS = "torch.nn.modules.loss._Loss"
TORCH_OPTIMIZERS = "torch.optim"
TORCH_OPTIMIZER = "torch.optim.Optimizer"
TORCH_SCHEDULERS = "torch.optim.lr_scheduler"
TORCH_SCHEDULER = "torch.optim.lr_scheduler.LRScheduler"


class _PytorchDeepAdapter:
    """Mixin adapter class for pytorch deep learning estimators."""

    _default_criterion = None
    _y_dtype = "float"

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        super().__dynamic_tags__()
        if self.metrics is not None:
            self.set_tags(**{"tests:python_dependencies": "torchmetrics"})

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        super().__post_init__()

        if self.random_state is not None:
            from torch import manual_seed

            manual_seed(self.random_state)

        activation_map = {}
        for var in self._instantiate_activation_vars:
            activation_map[var] = self._activation_spec(var)
        self._callable_activations = self._instantiate_activations(activation_map)
        self._metrics_objects = None

    def _activation_spec(self, var):
        """Get the activation to instantiate, for the parameter named ``var``."""
        return getattr(self, var, None)

    def _align_pred(self, y_pred, outputs):
        """Align the shape of the network output with the shape of the target."""
        return y_pred

    def _metric_kwargs(self, metric):
        """Get the kwargs to construct ``metric`` from ``torchmetrics`` with."""
        return {}

    def _run_epoch(self, epoch, dataloader):
        """Train the network for one epoch, and step the schedulers."""
        losses = []
        metric_values = {name: [] for name in (self._metrics_objects or {})}

        for inputs, outputs in dataloader:
            y_pred = self.network(**inputs)
            y_pred = self._align_pred(y_pred, outputs)
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
            from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        """Instantiate the activations of the estimator.

        Parameters
        ----------
        activations : dict of str to str, torch.nn.Module, or None
            The activations to instantiate, keyed by the name of the parameter
            they belong to. Values are the activation passed by the user, or the
            default of the estimator if the user passed none.

        Returns
        -------
        dict of str to torch.nn.Module or None
            The instantiated activations, keyed as in ``activations``.
            The value is None wherever the activation passed was None.
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

            # look up the activation of that name in torch.nn, case insensitively
            activation_class = _lookup_class(
                activation,
                module_path=TORCH_NN,
                base_class_path=TORCH_MODULE,
            )
            if activation_class is None:
                raise ValueError(
                    f"Activation '{activation}' is not a valid PyTorch activation"
                    "function in torch.nn module. Please pass a valid PyTorch"
                    "activation function in torch.nn module. Refer "
                    "https://pytorch.org/docs/stable/nn.html#non-linear-activations-"
                    "weighted-sum-nonlinearity for list of valid activation functions."
                )

            callable_activations[activation_var] = activation_class()
        return callable_activations

    def _instantiate_schedulers(self):
        """Instantiate the learning rate schedulers to be used during training.

        Currently, only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially,
        in the order in which they are passed.

        Only the names of schedulers are accepted, and not instances, since
        PyTorch learning rate schedulers must be constructed with the optimizer
        object, which does not exist before ``fit`` is called.

        Returns
        -------
        list of torch.optim.lr_scheduler.LRScheduler, or None
            The schedulers named in ``callbacks``, constructed on the optimizer,
            in the order in which they were passed.
            None if no callbacks were passed.
            For the schedulers that can be named, see
            https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        """
        if self.callbacks is None:
            return None

        if not isinstance(self.callbacks, tuple):
            self._callbacks = (self.callbacks,)
        else:
            self._callbacks = self.callbacks

        schedulers = []
        for scheduler in self._callbacks:
            if isinstance(scheduler, str):
                # look up the scheduler of that name in torch.optim.lr_scheduler,
                # case insensitively
                scheduler_class = _lookup_class(
                    scheduler,
                    module_path=TORCH_SCHEDULERS,
                    base_class_path=TORCH_SCHEDULER,
                )
                if scheduler_class is None:
                    all_callbacks = _lc_class_dict(TORCH_SCHEDULERS, TORCH_SCHEDULER)
                    raise ValueError(
                        f"Unknown learning rate scheduler: {scheduler}. "
                        f"Please pass one/many of {', '.join(sorted(all_callbacks))} "
                        "as a callback. Currently only learning rate schedulers are "
                        "supported as callbacks."
                    )
                if self.callback_kwargs:
                    schedulers.append(
                        scheduler_class(self._optimizer, **self.callback_kwargs)
                    )
                else:
                    schedulers.append(scheduler_class(self._optimizer))
            else:
                raise TypeError(
                    "Callbacks can either be None, a str or a tuple of str representing"
                    " a learning rate scheduler defined in PyTorch. "
                    "As currently only learning rate schedulers are "
                    f"supported as callbacks. But got {type(scheduler)} instead."
                )
        return schedulers

    def _instantiate_optimizer(self):
        """Instantiate the optimizer, on the parameters of the network.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer, constructed on ``self.network.parameters()``,
            with ``lr`` and ``optimizer_kwargs``.
        """
        from torch.optim import Adam, Optimizer

        # if no optimizer is passed, use Adam as default
        if self.optimizer is None:
            optimizer_class = Adam
            optimizer_params = {"lr": self.lr}
        # if optimizer is a string, look up the optimizer of that name
        # in torch.optim, case insensitively
        elif isinstance(self.optimizer, str):
            optimizer_class = _lookup_class(
                self.optimizer,
                module_path=TORCH_OPTIMIZERS,
                base_class_path=TORCH_OPTIMIZER,
            )
            if optimizer_class is None:
                all_optimizers = _lc_class_dict(TORCH_OPTIMIZERS, TORCH_OPTIMIZER)
                raise ValueError(
                    f"Unknown optimizer: {self.optimizer}. Please pass one of "
                    f"{', '.join(sorted(all_optimizers))} for `optimizer`."
                )
            optimizer_params = {"lr": self.lr}
        # if optimizer is an optimizer class, use it as is
        elif isinstance(self.optimizer, type) and issubclass(self.optimizer, Optimizer):
            optimizer_class = self.optimizer
            optimizer_params = {"lr": self.lr}
        # if optimizer is an instance of torch.optim.Optimizer, it cannot be used
        # directly: it is bound to the parameters it was constructed with, which
        # are never the parameters of self.network. Its hyperparameters are carried
        # over to a new optimizer of the same class, bound to the network instead.
        elif isinstance(self.optimizer, Optimizer):
            optimizer_class = type(self.optimizer)
            optimizer_params = dict(self.optimizer.defaults)
            # the learning rate of the instance is retained,
            # unless `lr` was explicitly set to a non-default value
            if self.lr != self.get_param_defaults().get("lr", self.lr):
                optimizer_params["lr"] = self.lr
        # if optimizer is neither None, nor a string, nor a class or instance of
        # a valid PyTorch optimizer, raise an error
        else:
            raise TypeError(
                "`optimizer` can either be None, a str, or a class or instance of "
                "optimizers defined in torch.optim. "
                "See https://pytorch.org/docs/stable/optim.html#algorithms. "
                f"But got {type(self.optimizer)} instead."
            )

        if self.optimizer_kwargs:
            optimizer_params.update(self.optimizer_kwargs)

        return optimizer_class(self.network.parameters(), **optimizer_params)

    def _instantiate_criterion(self):
        """Instantiate the loss function to train the network with.

        Returns
        -------
        torch.nn.modules.loss._Loss
            The loss function, constructed with ``criterion_kwargs``.
        """
        from torch import nn
        from torch.nn.modules.loss import _Loss

        if not self._validated_criterion:
            return getattr(nn, self._default_criterion)()
        # if criterion is a string, look up the loss function of that name
        # in torch.nn, case insensitively
        if isinstance(self._validated_criterion, str):
            criterion_class = _lookup_class(
                self._validated_criterion,
                module_path=TORCH_NN,
                base_class_path=TORCH_LOSS,
            )
            if criterion_class is None:
                all_criterions = _lc_class_dict(TORCH_NN, TORCH_LOSS)
                raise ValueError(
                    f"Unknown criterion: {self._validated_criterion}. Please pass one "
                    f"of {', '.join(sorted(all_criterions))} for `criterion`."
                )
            if self.criterion_kwargs:
                return criterion_class(**self.criterion_kwargs)
            else:
                return criterion_class()
        # if criterion is already an instance of torch.nn.modules.loss._Loss, use it
        elif isinstance(self._validated_criterion, _Loss):
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

    def _instantiate_metric(self, metric, torchmetrics):
        """Instantiate a single metric from torchmetrics.

        Parameters
        ----------
        metric : str or Callable
            Metric name from torchmetrics or a metric instance.
        torchmetrics : module
            The torchmetrics module.

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
                    f"Error in constructing torch based estimator "
                    f"{type(self).__name__}, "
                    f"unknown metric: {metric}. Please pass one of the available "
                    f"metrics from torchmetrics or check the metric name. "
                    f"See https://lightning.ai/docs/torchmetrics/stable/"
                )
            metric_class = getattr(torchmetrics, metric)
            return metric, metric_class(**self._metric_kwargs(metric))
        if isinstance(metric, Callable):
            return metric.__class__.__name__, metric
        raise TypeError(
            "`metrics` can either be None, a str or a tuple of str "
            "representing metrics from torchmetrics, or an instance of a "
            f"torchmetrics metric. But got {type(metric)} instead."
        )

    def _instantiate_metrics(self, metrics):
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
                metric, torchmetrics
            )
            metrics_dict[metric_name] = metric_instance

        return metrics_dict if metrics_dict else None

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        """Build the dataloader iterated over in fit and predict."""
        from torch.utils.data import DataLoader

        from sktime.base.adapters._pytorch_dataset import PytorchDataset

        dataset = PytorchDataset(X, y, self._y_dtype)
        return DataLoader(dataset, self.batch_size)

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch estimators.

        PyTorch estimators require 3D input and we don't allow automatic conversion
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch estimators require properly "
                f"formatted 3D time series data. Please reshape your data or "
                "use a supported Panel mtype."
            )

        # Call parent method for other conversions
        return super()._internal_convert(X, y)
