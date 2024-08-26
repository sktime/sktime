"""Pytorch Multivariate Time Series Transformer Model."""

import numpy as np

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class MVTSTransformerClassifier(BaseDeepClassifierPytorch):
    """Multivariate Time Series Transformer for Classification, as described in [1]_.

    This classifier has been wrapped around the official pytorch implementation of
    Transformer from [2]_, provided by the authors of the paper [1]_.

    References
    ----------
    .. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
    and Carsten Eickhoff. 2021. A Transformer-based Framework
    for Multivariate Time Series Representation Learning.
    In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery
    & Data Mining (KDD '21). Association for Computing Machinery, New York, NY, USA,
    2114-2124. https://doi.org/10.1145/3447548.3467401.
    .. [2] https://github.com/gzerveas/mvts_transformer

    Parameters
    ----------
    d_model : int, optional (default=256)
        The number of expected features in the input (i.e., the dimension of the model).
    n_heads : int, optional (default=4)
        The number of heads in the multihead attention mechanism.
    num_layers : int, optional (default=4)
        The number of layers (or blocks) in the transformer encoder.
    dim_feedforward : int, optional (default=128)
        The dimension of the feedforward network model.
    dropout : float, optional (default=0.1)
        The dropout rate to apply.
    pos_encoding : str, optional (default="fixed")
        The type of positional encoding to use. Options: ["fixed", "learnable"].
    activation : str, optional (default="relu")
        The activation function to use. Options: ["relu", "gelu"].
    norm : str, optional (default="BatchNorm")
        The type of normalization to use. Options: ["BatchNorm", "LayerNorm"].
    freeze : bool, optional (default=False)
        If True, the transformer layers will be frozen and not trained.
    num_epochs : int, optional (default=10)
        The number of epochs to train the model.
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : callable, optional (default=None)
        The loss function to use. If None, CrossEntropyLoss will be used.
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer : str, optional (default=None)
        The optimizer to use. If None, Adam optimizer will be used.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate for the optimizer.
    verbose : bool, optional (default=True)
        If True, prints progress messages during training.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.classification.deep_learning import MVTSTransformerClassifier
    >>>
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, _ = load_unit_test(split="test")
    >>>
    >>> model = MVTSTransformerClassifier()
    >>> model.fit(X_train, y_train)  # doctest: +SKIP
    >>> preds = model.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["gzerveas", "geetu040"],
        # gzerveas for original code in research repository
        "maintainers": ["geetu040"],
    }

    def __init__(
        self,
        # model specific
        d_model=256,
        n_heads=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        pos_encoding="fixed",
        activation="relu",
        norm="BatchNorm",
        freeze=False,
        # base classifier specific
        num_epochs=10,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        verbose=True,
        random_state=None,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.activation = activation
        self.norm = norm
        self.freeze = freeze
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # infer from the data
        self.feat_dim = None
        self.max_len = None
        self.num_classes = None

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

        from sktime.utils.dependencies import _check_soft_dependencies

        if _check_soft_dependencies("torch"):
            import torch

            self.criterions = {}

            self.optimizers = {
                "Adadelta": torch.optim.Adadelta,
                "Adagrad": torch.optim.Adagrad,
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "SGD": torch.optim.SGD,
            }

    def _build_network(self, X, y):
        from sktime.networks.mvts_transformer import (
            TSTransformerEncoderClassiregressor,
        )

        # n_instances, n_dims, n_timestamps
        _, self.feat_dim, self.max_len = X.shape

        self.num_classes = len(np.unique(y))

        return TSTransformerEncoderClassiregressor(
            feat_dim=self.feat_dim,
            max_len=self.max_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            num_classes=self.num_classes,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            activation=self.activation,
            norm=self.norm,
            freeze=self.freeze,
        )

    def _build_dataloader(self, X, y=None):
        dataset = PytorchDataset(X, y)
        return DataLoader(dataset, self.batch_size)

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
        params = [
            {
                "d_model": 16,
                "n_heads": 1,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0,
                "pos_encoding": "fixed",
                "activation": "relu",
                "norm": "BatchNorm",
                "freeze": False,
                "num_epochs": 1,
                "verbose": False,
                "random_state": 0,
            },
            {
                "d_model": 16,
                "n_heads": 1,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0,
                "pos_encoding": "learnable",
                "activation": "gelu",
                "norm": "LayerNorm",
                "freeze": True,
                "num_epochs": 1,
                "verbose": False,
                "random_state": 0,
            },
        ]
        return params


class PytorchDataset(Dataset):
    """Dataset specifc to TransformerClassifier."""

    def __init__(self, X, y):
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
        x = self.X[i]
        x = torch.tensor(x, dtype=torch.float)
        padding_masks = torch.ones(x.shape[:-1], dtype=torch.bool)

        inputs = {
            "X": x,
            "padding_masks": padding_masks,
        }

        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torch.tensor(y, dtype=torch.long)
        return inputs, y
