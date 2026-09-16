"""Abstract base class for the PyTorch neural network regressors."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepRegressorTorch"]

from collections.abc import Callable

from sktime.base.adapters._pytorch import _PytorchDeepAdapter
from sktime.regression.base import BaseRegressor
from sktime.utils.dependencies import _safe_import


class BaseDeepRegressorTorch(_PytorchDeepAdapter, BaseRegressor):
    """Abstract base class for the PyTorch neural network regressors.

    Parameters
    ----------
    num_epochs : int, default = 16
        The number of epochs to train the model
    batch_size : int, default = 8
        The size of each mini-batch during training
    criterion : case insensitive str, or an instance of a loss function
        defined in PyTorch, default = None
        The loss function to use for training the model.
        If None, MSELoss is used.

        Permitted values:

        - ``None``: the ``MSELoss`` loss function is used.
        - ``str``: case insensitive name of any loss function in ``torch.nn``,
          for example ``"l1loss"`` or ``"L1Loss"``. See
          https://pytorch.org/docs/stable/nn.html#loss-functions
        - instance of a loss function defined in ``torch.nn``, for example
          ``torch.nn.L1Loss()``. An instance is used as is.

        If a str is passed, the loss function is constructed with
        ``criterion_kwargs``.
    criterion_kwargs : dict, default = None
        The keyword arguments to be passed to the loss function.
    optimizer : case insensitive str, or an instance of an optimizer
        defined in PyTorch, default = None
        The optimizer to use for training the model.
        If None, Adam is used.

        Permitted values:

        - ``None``: the ``Adam`` optimizer is used.
        - ``str``: case insensitive name of any optimizer in ``torch.optim``,
          for example ``"adam"`` or ``"SGD"``. See
          https://pytorch.org/docs/stable/optim.html#algorithms
        - instance of a subclass of ``torch.optim.Optimizer``, for example
          ``torch.optim.SGD(model.parameters(), lr=0.01)``.

        If a str is passed, the optimizer is constructed on the parameters of the
        network, with ``lr`` and ``optimizer_kwargs``.
    optimizer_kwargs : dict, default = None
        The keyword arguments to be passed to the optimizer.
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
        Examples: "MeanSquaredError", "MeanAbsoluteError", "R2Score"
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
        "tests:libs": ["sktime.regression.deep_learning.base._base_torch"],
    }

    # _instantiate_activation_vars is an iterable of attribute names of activations
    # to instantiate. In case activation attributes in subclasses are different than
    # the default ones (activation and activation_hidden), this variable should
    # be overridden.
    _instantiate_activation_vars = ("activation", "activation_hidden")

    _default_criterion = "torch.nn.MSELoss"

    @property
    def _validated_criterion(self):
        return self.criterion

    def _align_pred(self, y_pred, outputs):
        if (
            y_pred.ndim == 2
            and y_pred.shape[1] == 1
            and outputs.ndim == 1
            and y_pred.shape[0] == outputs.shape[0]
        ):
            return y_pred.squeeze(-1)
        return y_pred

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
        metrics: None | str | Callable | tuple[str | Callable, ...] = None,
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
        self.metrics = metrics
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self.network = self._build_network(X)

        # instantiate loss function and optimizer
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        # instantiate callbacks (learning rate schedulers)
        self._schedulers = self._instantiate_schedulers()
        # instantiate metrics
        self._metrics_objects = self._instantiate_metrics(self.metrics)
        # build dataloader
        dataloader = self._build_dataloader(X, y)

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

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

        # (n_instances, 1) -> (n_instances,), also for n_instances == 1
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(-1)
        return y_pred
