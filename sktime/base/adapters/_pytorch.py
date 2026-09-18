"""Implements adapter for pytorch deep learning estimators."""

__all__ = ["_PytorchDeepAdapter", "PytorchDataset"]
__authors__ = ["geetu040", "RecreationalMath"]

import abc
from collections.abc import Callable

import numpy as np

from sktime.utils._lookup import _lc_class_dict, _lookup_class
from sktime.utils.dependencies import _safe_import

ReduceLROnPlateau = _safe_import("torch.optim.lr_scheduler.ReduceLROnPlateau")

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
    _y_dtype = "torch.float"

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.metrics is not None:
            self.set_tags(**{"tests:python_dependencies": "torchmetrics"})

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        # set random seed for torch
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

        activation_map = {}
        for var in self._instantiate_activation_vars:
            activation_map[var] = self._activation_spec(var)
        self._callable_activations = self._instantiate_activations(activation_map)
        self._metrics_objects = None

    def _activation_spec(self, var):
        return getattr(self, var, None)

    def _align_pred(self, y_pred, outputs):
        return y_pred

    def _metric_kwargs(self, metric):
        return {}

    def _run_epoch(self, epoch, dataloader):
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
        """Instantiate the schedulers to be used during training.

        Currently, only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially
        in the order they are passed.

        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.

        Sets
        ------
        self._schedulers : None or str or a tuple of str, each string
            representing the name of a valid learning rate scheduler
            implemented in PyTorch. For list of supported learning rate schedulers
            see: https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            The list of instantiated schedulers to be used during training.
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
        # import the base class for all optimizers in PyTorch
        torchOptimizer = _safe_import(TORCH_OPTIMIZER)

        # if no optimizer is passed, use Adam as default
        if self.optimizer is None:
            optimizer_class = _safe_import("torch.optim.Adam")
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
        elif isinstance(self.optimizer, type) and issubclass(
            self.optimizer, torchOptimizer
        ):
            optimizer_class = self.optimizer
            optimizer_params = {"lr": self.lr}
        # if optimizer is an instance of torch.optim.Optimizer, it cannot be used
        # directly: it is bound to the parameters it was constructed with, which
        # are never the parameters of self.network. Its hyperparameters are carried
        # over to a new optimizer of the same class, bound to the network instead.
        elif isinstance(self.optimizer, torchOptimizer):
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
        if not self._validated_criterion:
            return _safe_import(self._default_criterion)()
        # import the base class for all loss functions in PyTorch
        torchLossFunction = _safe_import(TORCH_LOSS)
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
        dataset = PytorchDataset(X, y, self._y_dtype)
        DataLoader = _safe_import("torch.utils.data.DataLoader")
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for estimators:
                "results_comparison" - used for identity testing in some estimators
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
    """Dataset for use in sktime deep learning estimators based on pytorch."""

    def __init__(self, X, y=None, y_dtype="torch.float"):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y
        self.y_dtype = y_dtype

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        torchTensor = _safe_import("torch.tensor")
        torchFloat = _safe_import("torch.float")
        x = self.X[i]
        x = torchTensor(x, dtype=torchFloat)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torchTensor(y, dtype=_safe_import(self.y_dtype))
        return inputs, y
