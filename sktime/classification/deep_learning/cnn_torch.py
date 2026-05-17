"""Time Convolutional Neural Network (CNN) classifier - PyTorch implementation."""

__author__ = ["hfawaz", "James-Large", "noxthot", "Aanoush-Surana"]
__all__ = ["CNNClassifierTorch"]

from sktime.classification.deep_learning.base._base_torch import (
    BaseDeepClassifierPytorch,
)


class CNNClassifierTorch(BaseDeepClassifierPytorch):
    """Time Convolutional Neural Network (CNN), as described in [1]_.

    PyTorch implementation of CNNClassifier.

    Parameters
    ----------
    kernel_size : int, default=7
        The length of the 1D convolution window.
    avg_pool_size : int, default=3
        Size of the average pooling windows.
    n_conv_layers : int, default=2
        The number of convolutional plus average pooling layers.
    filter_sizes : list of int, default=None
        Number of filters per conv layer. Defaults to [6, 12].
    num_epochs : int, default=2000
        The number of epochs to train the model.
    batch_size : int, default=16
        The number of samples per gradient update.
    criterion : callable or None, default=None
        Loss function. If None, CrossEntropyLoss is used.
    criterion_kwargs : dict or None, default=None
        Keyword arguments for the loss function.
    optimizer : str or None, default=None
        Optimizer name. If None, Adam is used.
    optimizer_kwargs : dict or None, default=None
        Keyword arguments for the optimizer.
    lr : float, default=0.001
        Learning rate.
    verbose : bool, default=False
        Whether to print training progress.
    random_state : int or None, default=None
        Seed for reproducibility.

    References
    ----------
    .. [1] Zhao et al., Convolutional neural networks for time series
       classification, Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from sktime.classification.deep_learning.cnn_torch import CNNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> clf = CNNClassifierPytorch(num_epochs=5, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    CNNClassifierTorch(...)
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "noxthot", "Aanoush-Surana"],
        "maintainers": ["Aanoush-Surana"],
        "python_dependencies": ["torch"],
        "capability:multivariate": True,
        "capability:random_state": True,
    }

    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        num_epochs=2000,
        batch_size=16,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        verbose=False,
        random_state=None,
    ):
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        if filter_sizes is None:
            self.filter_sizes = [6, 12][:n_conv_layers]
        else:
            self.filter_sizes = filter_sizes

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

    def _build_network(self, X, y):
        import torch.nn as nn

        n_channels = X.shape[1]
        n_classes = self.n_classes_
        kernel_size = self.kernel_size
        avg_pool_size = self.avg_pool_size
        filter_sizes = self.filter_sizes

        class _CNNNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.pools = nn.ModuleList()
                self.bns = nn.ModuleList()

                in_channels = n_channels
                for i in range(len(filter_sizes)):
                    self.convs.append(
                        nn.Conv1d(
                            in_channels,
                            filter_sizes[i],
                            kernel_size=kernel_size,
                            padding="same",
                            bias=True,
                        )
                    )
                    self.bns.append(nn.BatchNorm1d(filter_sizes[i]))
                    self.pools.append(nn.AvgPool1d(kernel_size=avg_pool_size))
                    in_channels = filter_sizes[i]

                self.relu = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc = None
                self._in_channels = n_channels
                self._n_classes = n_classes

            def forward(self, x):
                x = x.permute(0, 2, 1)

                for conv, bn, pool in zip(self.convs, self.bns, self.pools):
                    x = self.relu(bn(conv(x)))
                    x = pool(x)

                x = self.flatten(x)

                if self.fc is None:
                    self.fc = nn.Linear(x.shape[1], self._n_classes).to(x.device)

                return self.fc(x)

        return _CNNNetwork()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "num_epochs": 1,
            "batch_size": 4,
            "avg_pool_size": 2,
            "kernel_size": 3,
            "filter_sizes": [4, 8],
            "lr": 1e-3,
            "random_state": 0,
        }
        params2 = {
            "num_epochs": 2,
            "batch_size": 8,
            "n_conv_layers": 1,
            "filter_sizes": [6],
            "optimizer": "Adam",
            "lr": 5e-4,
            "random_state": 42,
        }
        return [params1, params2]
