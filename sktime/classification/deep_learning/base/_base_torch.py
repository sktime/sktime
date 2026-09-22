"""Abstract base class for the Pytorch neural network classifiers."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepClassifierPytorch"]

from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import LabelEncoder

from sktime.base.adapters._pytorch import _PytorchDeepAdapter
from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _safe_import


class BaseDeepClassifierPytorch(_PytorchDeepAdapter, BaseClassifier):
    """Abstract base class for the Pytorch neural network classifiers.

    Parameters
    ----------
    num_epochs : int, default = 16
        The number of epochs to train the model
    batch_size : int, default = 8
        The size of each mini-batch during training
    activation : case insensitive str, or an instance of an activation
        defined in PyTorch, default = None
        The activation applied to the output layer.
        If None, no activation is applied.

        Permitted values:

        - ``None``: no activation is applied to the output layer and the network
          returns raw outputs (logits). This is typically required when using
          ``CrossEntropyLoss``, which expects logits as input.
        - ``str``: case insensitive name of any activation in ``torch.nn``,
          for example ``"relu"`` or ``"LeakyReLU"``. See
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - instance of a subclass of ``torch.nn.Module``, for example
          ``torch.nn.ReLU()``. Arbitrary callables are not supported.

        If a str is passed, the activation is constructed with default arguments.
    criterion : case insensitive str, or an instance of a loss function
        defined in PyTorch, default = None
        The loss function to use for training the model.
        If None, CrossEntropyLoss is used.

        Permitted values:

        - ``None``: the ``CrossEntropyLoss`` loss function is used.
        - ``str``: case insensitive name of any loss function in ``torch.nn``,
          for example ``"nllloss"`` or ``"NLLLoss"``. See
          https://pytorch.org/docs/stable/nn.html#loss-functions
        - instance of a loss function defined in ``torch.nn``, for example
          ``torch.nn.NLLLoss()``. An instance is used as is.

        If a str is passed, the loss function is constructed with
        ``criterion_kwargs``.
    criterion_kwargs : dict or None, default = None
        The keyword arguments to be passed to the loss function.
    optimizer : case insensitive str, or a class or instance of an optimizer
        defined in PyTorch, default = None
        The optimizer to use for training the model. If None, Adam optimizer is used.

        Permitted values:

        - ``None``: the ``Adam`` optimizer is used.
        - ``str``: case insensitive name of any optimizer in ``torch.optim``,
          for example ``"adam"`` or ``"SGD"``. See
          https://pytorch.org/docs/stable/optim.html#algorithms
        - ``class``: a subclass of ``torch.optim.Optimizer``, for example
          ``torch.optim.SGD``.
        - instance of a subclass of ``torch.optim.Optimizer``, for example
          ``torch.optim.SGD(model.parameters(), lr=0.01)``.

        In all cases the optimizer is constructed on the parameters of the network,
        with ``lr`` and ``optimizer_kwargs``.
    optimizer_kwargs : dict or None, default = None
        The keyword arguments to be passed to the optimizer. These take precedence
        over ``lr``, and over the hyperparameters of an ``optimizer`` instance.
    callbacks : case insensitive str, or a tuple of str, default = None
        The learning rate schedulers to use during training.
        Currently only learning rate schedulers are supported as callbacks.
        If None, no learning rate scheduler is used.

        Permitted values:

        - ``None``: no learning rate scheduler is used.
        - ``str``: case insensitive name of any learning rate scheduler in
          ``torch.optim.lr_scheduler``, for example ``"steplr"`` or ``"StepLR"``.
          See https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        - tuple of ``str``: multiple schedulers, which are applied sequentially,
          in the order in which they are passed.

        Schedulers are constructed with ``callback_kwargs``. Instances are not
        accepted, since PyTorch schedulers must be constructed with the optimizer,
        which does not exist before ``fit`` is called.
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
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
        # CI and test tags
        # ----------------
        "tests:vm": True,
        "tests:libs": ["sktime.classification.deep_learning.base._base_torch"],
    }

    # _instantiate_activation_vars is an iterable of attribute names of activations
    # to instantiate. In case activation attributes in subclasses are different than
    # the default ones (activation and activation_hidden), this variable should
    # be overridden.
    _instantiate_activation_vars = ("activation", "activation_hidden")

    _default_criterion = "CrossEntropyLoss"
    _y_dtype = "long"

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes."""
        self._validate_activation_criterion()
        super().__post_init__()
        self.label_encoder = None

    def _activation_spec(self, var):
        if var == "activation":
            return self._validated_activation
        return getattr(self, var, None)

    def _metric_kwargs(self, metric):
        kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        if metric in ("F1Score", "Precision", "Recall"):
            kwargs["average"] = "macro"
        return kwargs

    def __init__(
        self: "BaseDeepClassifierPytorch",
        num_epochs: int = 16,
        batch_size: int = 8,
        activation: str | None | Callable = None,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict | None = None,
        optimizer: str | Callable | None = None,
        optimizer_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = None,
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
        self.callback_kwargs = callback_kwargs
        self.metrics = metrics
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        if self.random_state is not None:
            import torch

            torch.manual_seed(self.random_state)

        y = self._encode_y(y)

        self.network = self._build_network(X, y)

        # instantiate loss function and optimizer
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        # instantiate callbacks (learning rate schedulers)
        self._schedulers = self._instantiate_schedulers()
        # ensure num_classes is set before instantiating metrics
        # as classification metrics require num_classes as an argument
        self.num_classes = len(np.unique(y))
        # instantiate metrics
        self._metrics_objects = self._instantiate_metrics(self.metrics)
        # build dataloader
        dataloader = self._build_dataloader(X, y)

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

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
            or 2D iterable, of shape [n_instances, n_outputs]
            predicted class labels
            indices correspond to instance indices in X
            if self.get_tag("capability:multioutput") = False, should be 1D
            if self.get_tag("capability:multioutput") = True, should be 2D
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
        y : 2D array of shape [n_instances, n_outputs] - predicted class probabilities
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
