"""Pytorch Transformer Model."""

import numpy as np

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class TransformerClassifier(BaseDeepClassifierPytorch):
    """Transformer for Classification, as described in [1]_.

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
    """

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
    }

    def __init__(
        self,
        # model specific
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
        # base classifier specific
        num_epochs=16,
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
        from sktime.networks.transformer_classifier import (
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
                special parameters are defined for a value,
                will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
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
