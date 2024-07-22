"""Pytorch Transformer Model."""

import numpy as np

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import DataLoader, Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class TransformerClassifier(BaseDeepClassifierPytorch):
    """Transformer Classifier based on Pytorch."""

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
        from sktime.classification.deep_learning.transformer.network import (
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
        from sktime.classification.deep_learning.transformer.dataset import (
            PytorchDataset,
        )

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
