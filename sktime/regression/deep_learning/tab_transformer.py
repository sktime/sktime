"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

import numpy as np

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""


__author__ = ["Ankit-1204"]


class Tab_Transformer(NNModule):
    r"""Tab Transformer Architecture.

    Parameters
    ----------
    num_cat_feat : list,
        array of unique classes for each categorical feature
    num_cont_features : int,
        number of continuos features
    embedding dim : int,
        value of embedding dimension
    n_transformer_layer : int,
        number of transformer layers
    n_heads : int,
        number of attention heads
    output_dim : int,
        dimension for network output
    task : string,
        classification or regression task.

    """

    def __init__(
        self,
        num_cat_feat,
        num_cont_features,
        embedding_dim,
        n_transformer_layer,
        n_heads,
        output_dim,
        task="classification",
    ) -> None:
        super().__init__()
        self.num_cat_feat = num_cat_feat
        self.num_cont_features = num_cont_features
        self.embedding_dim = embedding_dim
        self.n_transformer_layer = n_transformer_layer
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.task = task
        self.embedding = nn.ModuleList(
            [nn.Embedding(cat, self.embedding_dim) for cat in self.num_cat_feat]
        )
        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=self.n_heads
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, num_layers=self.n_transformer_layer
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                self.embedding_dim * len(self.num_cat_feat) + self.num_cont_features,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )
        if self.task == "classification":
            self.activation = nn.Softmax() if self.output_dim > 1 else nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x_cat, x_cont):
        """Implement forward for Tab Transformer."""
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding)]
        embedded = torch.stack(dim=1)

        context = self.transformer(embedded)
        context = context.view(context.size(0), -1)
        combined = torch.concat([context, x_cont], dim=1)
        feed = self.feed_forward(combined)
        return self.activation(feed)


class TabTransformer(BaseDeepNetworkPyTorch):
    r"""
    Tab Transformer for Tabular Data.

    Parameters
    ----------
    num_cat_feat : list,
        array of unique classes for each categorical feature
    num_cont_features : int,
        number of continuos features
    embedding dim : int,
        value of embedding dimension
    n_transformer_layer : int,
        number of transformer layers
    n_heads : int,
        number of attention heads
    output_dim : int,
        dimension for network output
    task : string,
        classification or regression task.

    References
    ----------
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
    Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
    https://arxiv.org/pdf/2012.06678
    """

    def __init__(
        self,
        num_cat_feat,
        num_cont_features,
        embedding_dim,
        n_transformer_layer,
        n_heads,
        output_dim,
        task="classification",
        num_epochs=16,
        batch_size=8,
        lr=0.001,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
    ):
        self.num_cat_feat = num_cat_feat
        self.num_cont_features = num_cont_features
        self.embedding_dim = embedding_dim
        self.n_transformer_layer = n_transformer_layer
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.task = task
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.lr = lr
        super().__init__()
        if _check_soft_dependencies("torch", severity="none"):
            pass

    def _window_classification(self, X, y, window_size=3):
        r"""Convert Timeseries into Tabular Format.

        Parameters
        ----------
        window_size : int, Optional (default=3)
            Defines the size of the sliding window
        X : 3d numpy array
            shape : (n_instances, series_length, n_dimensions)
        y : Required for Classification

        """
        x_transf = []
        y_transf = []
        for sample_id in range(len(X)):
            for i in range(X.shape[1] - window_size):
                x_curr = X[sample_id, i : i + window_size, :].flatten()
                y_transf.append(y[sample_id])
                x_transf.append(x_curr)
        x_transf = np.array(x_transf)
        y_transf = np.array(y_transf)

        return x_transf, y_transf

    def _window_regression(self, X, window_size=3):
        r"""Convert Timeseries into Tabular Format.

        Parameters
        ----------
        window_size : int, Optional (default=3)
            Defines the size of the sliding window
        X : 3d numpy array
            shape : (n_instances, series_length, n_dimensions)

        """
        x_transf = []
        y_transf = []
        for sample_id in range(len(X)):
            for i in range(X.shape[1] - window_size - 1):
                x_transf.append(X[sample_id, i : i + window_size, :].flatten())
                y_transf.append(X[sample_id, i + window_size, :])
        x_transf = np.array(x_transf)
        y_transf = np.array(y_transf)

        return x_transf, y_transf

    def _fit(self, X, y):
        r"""Fit the model on input.

        Parameters
        ----------
        X : 2d numpy array
            shape : (n_instances, series_length * n_dimensions)
        y : 1d or 2d numpy array
            for classification, shape = (n_instances)
            for regression, shape = (n_instances, n_dimensions)
        """
        dataloader = self._build_network(X, y)
        dataloader
        return
