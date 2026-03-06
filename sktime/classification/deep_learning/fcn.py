"""Fully Convolutional Network (FCN) for classification."""

__all__ = ["FCNClassifier"]

from sktime.classification.deep_learning.base._base_torch import (
    BaseDeepClassifierPytorch,
)


class FCNClassifier(BaseDeepClassifierPytorch):
    """Fully Convolutional Network (FCN), as described in [1]_.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    Parameters
    ----------
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
    .. [1] Wang et al, Time series classification from scratch with
    deep neural networks: A strong baseline.
    2017 International Joint Conference on Neural Networks (IJCNN)

    Examples
    --------
    >>> from sktime.classification.deep_learning.fcn import FCNClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> fcn = FCNClassifier(num_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> fcn.fit(X_train, y_train)  # doctest: +SKIP
    FCNClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "hfawaz",
            "James-Large",
            "AurumnPegasus",
            "noxthot",
            "Aanoush-Surana",
        ],
        # hfawaz for dl-4-tsc
        "maintainers": ["James-Large", "AurumnPegasus"],
        # estimator type handled by parent class
        "python_dependencies": ["torch"],
        "capability:multivariate": True,
        "capability:random_state": True,
    }

    def __init__(
        self,
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

        class _GlobalAvgPool(nn.Module):
            def forward(self, x):
                return x.mean(dim=-1)

        backbone = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=8, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            _GlobalAvgPool(),
            nn.Linear(128, n_classes),
        )

        class _FCNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = backbone

            def forward(self, X):
                # convert (batch, length, channels) -> (batch, channels, length)
                X = X.permute(0, 2, 1)
                return self.model(X)

        return _FCNNet()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {
            "num_epochs": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "random_state": 0,
        }
        params2 = {
            "num_epochs": 2,
            "batch_size": 8,
            "optimizer": "Adam",
            "lr": 5e-4,
            "random_state": 42,
        }
        return [params1, params2]
