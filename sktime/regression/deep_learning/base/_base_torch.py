"""Abstract base class for the PyTorch neural network regressors."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepRegressorTorch"]

import abc
from collections.abc import Callable

import numpy as np

from sktime.regression.base import BaseRegressor
from sktime.utils.dependencies import _safe_import

ReduceLROnPlateau = _safe_import("torch.optim.lr_scheduler.ReduceLROnPlateau")


class BaseDeepRegressorTorch(BaseRegressor):
    """Abstract base class for the PyTorch neural network regressors.

    Parameters
    ----------
    num_epochs : int, default = 16
        The number of epochs to train the model
    batch_size : int, default = 8
        The size of each mini-batch during training
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
        self: "BaseDeepRegressorTorch",
        num_epochs: int = 16,
        batch_size: int = 8,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict = None,
        optimizer: str | Callable | None = None,
        optimizer_kwargs: dict = None,
        callbacks: None | str | tuple[str, ...] = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = True,
        random_state: int | None = None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

        # set random seed for torch
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

        # optimizers, criterions, callbacks will be instantiated in
        # _instantiate_optimizer, _instantiate_criterion & _instantiate_callbacks
        # methods respectively
        self._all_optimizers = None
        self._all_criterions = None
        self._all_callbacks = None

    def _fit(self, X, y):
        self.network = self._build_network(X)

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
        # print loss for the epoch, if verbose is True
        if self.verbose:
            print(f"Epoch {epoch + 1}: Loss: {epoch_loss}")

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
        # if no criterion is passed, use MSELoss as default
        if not self.criterion:
            loss = _safe_import("torch.nn.MSELoss")()
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
        if isinstance(self.criterion, str):
            if self.criterion.lower() in self._all_criterions:
                criterion_class = _safe_import(
                    f"torch.nn.{self._all_criterions[self.criterion.lower()]}"
                )
                if self.criterion_kwargs:
                    return criterion_class(**self.criterion_kwargs)
                else:
                    return criterion_class()
            else:
                raise ValueError(
                    f"Unknown criterion: {self.criterion}. Please pass one "
                    f"of {', '.join(self._all_criterions)} for `criterion`."
                )
        # if criterion is already an instance of torch.nn.modules.loss._Loss, use it
        elif isinstance(self.criterion, torchLossFunction):
            return self.criterion
        else:
            # if criterion is neither a string nor an instance of
            # a valid PyTorch loss function, raise an error
            raise TypeError(
                "`criterion` can either be None, a str or an instance of "
                "loss functions defined in "
                "https://pytorch.org/docs/stable/nn.html#loss-functions "
                f"But got {type(self.criterion)} instead."
            )

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        # default behaviour if estimator does not implement
        # dataloader of its own
        dataset = PytorchDataset(X, y)
        DataLoader = _safe_import("torch.utils.data.DataLoader")
        return DataLoader(dataset, self.batch_size)

    def _predict(self, X):
        """Predict target for sequences in X.

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
            predicted values
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
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
        y_pred = y_pred.numpy()

        # For single-output regression, squeeze if needed
        # if y_pred has shape (n_instances, 1), convert to (n_instances,)
        # to conform to expected output shape
        # (n_instances, 1) -> (n_instances,)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze()
        return y_pred

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch regressors.

        PyTorch regressors require 3D input and we don't allow automatic conversion
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch regressors require properly "
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
            Reserved values for regressors:
                "results_comparison" - used for identity testing in some regressors
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
    """Dataset for use in sktime deep learning regressor based on pytorch."""

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
        torchFloat = _safe_import("torch.float")
        x = self.X[i]
        x = torchTensor(x, dtype=torchFloat)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torchTensor(y, dtype=torchFloat)
        return inputs, y
