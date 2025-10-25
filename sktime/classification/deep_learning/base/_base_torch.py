"""Abstract base class for the Pytorch neural network classifiers."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepClassifierPytorch"]

import abc
from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import LabelEncoder

from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _safe_import

ReduceLROnPlateau = _safe_import("torch.optim.lr_scheduler.ReduceLROnPlateau")


class BaseDeepClassifierPytorch(BaseClassifier):
    """Abstract base class for the Pytorch neural network classifiers.

    Parameters
    ----------
    num_epochs : int, default = 100
        The number of epochs to train the model
    batch_size : int, default = 8
        The size of each mini-batch during training
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer.
    criterion : case insensitive str or an instance of a loss function
        defined in PyTorch, default = None
        The loss function to be used in training the neural network.
        If None, CrossEntropyLoss is used.
        If a string/Callable is passed, it must be one of the loss functions defined in
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict, default = None
        The keyword arguments to be passed to the loss function.
    optimizer : case insensitive str or an instance of an optimizer
        defined in PyTorch, default = None
        The optimizer to use for training the model. If None, Adam optimizer is used.
        If a string/Callable is passed, it must be one of the optimizers defined in
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict, default = None
        The keyword arguments to be passed to the optimizer.
    callbacks : None or str or a tuple of str, default = None
        Currently only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially in the
        order they are passed. If None, then no learning rate scheduler is used.
        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.
        List of available learning rate schedulers:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
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
    }

    def __init__(
        self: "BaseDeepClassifierPytorch",
        num_epochs: int = 16,
        batch_size: int = 8,
        activation: str | None | Callable = None,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict = None,
        optimizer: str | Callable | None = None,
        optimizer_kwargs: dict = None,
        callbacks: None | Callable | tuple[Callable, ...] = None,
        callback_kwargs: dict | None = None,
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
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # use this when y has str
        self.label_encoder = None
        super().__init__()

        # set random seed for torch
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

        # validate activation function w.r.t. criterion specified
        self._validate_activation_criterion()
        # post this function call,
        # self.validated_criterion and self.validated_activation are used
        # and self.criterion and self.activation are ignored

        # optimizers, criterions, callbacks will be instantiated in
        # _instantiate_optimizer, _instantiate_criterion & _instantiate_callbacks
        # methods respectively
        self._all_optimizers = None
        self._all_criterions = None
        self._all_callbacks = None

    def _fit(self, X, y):
        y = self._encode_y(y)

        self.network = self._build_network(X, y)

        # instantiate loss function and optimizer
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        # instantiate callbacks (learning rate schedulers)
        self._schedulers = self._instantiate_schedulers()
        # build dataloader
        dataloader = self._build_dataloader(X, y)

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _run_epoch(self, epoch, dataloader):
        losses = []
        for inputs, outputs in dataloader:
            y_pred = self.network(**inputs)
            loss = self._criterion(y_pred, outputs)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
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
        # print loss for the epoch
        if self.verbose:
            print(f"Epoch {epoch + 1}: Loss: {epoch_loss}")

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
        self.validated_criterion : str or Callable
            The validated criterion to be used in training the neural network.
            This will either be the same as self.criterion, or "crossentropyloss"
            if a functionally equivalent combination of criterion and activation
            function is detected.
        self.validated_activation : str or Callable or None
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
        for scheduler in self._callbacks:
            if isinstance(scheduler, str):
                if scheduler.lower() in self._all_callbacks:
                    scheduler_class = _safe_import(
                        f"torch.optim.lr_scheduler.{self._all_callbacks[scheduler.lower()]}"  # noqa: E501
                    )
                    if self.callback_kwargs:
                        schedulers.append(
                            scheduler_class(self._optimizer, **self.callback_kwargs)
                        )
                    else:
                        schedulers.append(scheduler_class(self._optimizer))
                else:
                    raise ValueError(
                        f"Unknown learning rate scheduler: {scheduler}. "
                        f"Please pass one/many of {', '.join(self._all_callbacks)} "
                        "as a callback. Currently only learning rate schedulers are "
                        "supported as callbacks."
                    )
            else:
                raise TypeError(
                    "Callbacks can either be None, a str or a tuple of str representing"
                    " a learning rate scheduler defined in PyTorch. "
                    "As currently only learning rate schedulers are "
                    f"supported as callbacks. But got {type(scheduler)} instead."
                )
        return schedulers

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
                if self.callback_kwargs:
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
        with torchNo_grad():
            for inputs in dataloader:
                y_pred.append(self.network(**inputs).detach())
        y_pred = cat(y_pred, dim=0)
        # (batch_size, num_outputs)
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
