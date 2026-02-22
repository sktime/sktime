"""ResNet regressor for time series in PyTorch."""

__authors__ = ["DCchoudhury15"]

__all__ = ["ResNetRegressorTorch"]

from collections.abc import Callable

from sktime.networks.resnet import ResNetNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class ResNetRegressorTorch(BaseDeepRegressorTorch):
    """Residual Network (ResNet) for time series regression in PyTorch.

    Parameters
    ----------
    n_feature_maps : int, default = 64
        Number of feature maps in the first residual block.
        Subsequent blocks use n_feature_maps * 2.
    activation_hidden : str, default = "relu"
        Activation function used in the residual blocks.
    num_epochs : int, default = 1500
        Number of epochs to train the model.
    batch_size : int, default = 16
        Number of samples per gradient update.
    optimizer : str or None or instance of torch.optim optimizer, default = "Adam"
        Optimizer to use for training.
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments for the optimizer.
    criterion : str or None or instance of torch loss, default = "MSELoss"
        Loss function.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments for the loss function.
    callbacks : str or None or tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers to use during training.
    callback_kwargs : dict or None, default = None
        Additional keyword arguments for the callbacks.
    lr : float, default = 0.01
        Learning rate for the optimizer.
    verbose : bool, default = False
        Whether to print progress during training.
    random_state : int or None, default = None
        Seed for reproducibility.

    References
    ----------
    .. [1] Wang et al, Time series classification from scratch with deep
        neural networks: A strong baseline, IJCNN 2017.

    Examples
    --------
    >>> from sktime.regression.deep_learning.resnet import ResNetRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> reg = ResNetRegressorTorch(num_epochs=10, batch_size=4) # doctest: +SKIP
    >>> reg.fit(X_train, y_train) # doctest: +SKIP
    ResNetRegressorTorch(...)
    """

    _tags = {
        "authors": ["DCchoudhury15"],
        "maintainers": ["DCchoudhury15"],
        "python_dependencies": "torch",
        "capability:multivariate": True,
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        n_feature_maps: int = 64,
        activation_hidden: str = "relu",
        num_epochs: int = 1500,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "MSELoss",
        criterion_kwargs: dict | None = None,
        callbacks: str | None | tuple = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.n_feature_maps = n_feature_maps
        self.activation_hidden = activation_hidden

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            callbacks=callbacks,
            callback_kwargs=callback_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

    def _build_network(self, X):
        """Build the ResNet network.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, series_length)
            Training input data.

        Returns
        -------
        network : ResNetNetworkTorch instance
            The constructed ResNet network.
        """
        _, input_size, _ = X.shape

        return ResNetNetworkTorch(
            input_size=input_size,
            num_classes=1,
            n_feature_maps=self.n_feature_maps,
            activation_hidden=self.activation_hidden,
            random_state=self.random_state,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "num_epochs": 3,
            "batch_size": 4,
            "n_feature_maps": 16,
            "callbacks": None,
        }
        params2 = {
            "num_epochs": 5,
            "batch_size": 8,
            "n_feature_maps": 32,
            "lr": 0.001,
            "callbacks": None,
            "random_state": 0,
        }
        return [params1, params2]
