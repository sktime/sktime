"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.utils.data import Dataset

    DataSet = Dataset
    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""

    class DataSet:
        """Dummy class if torch is unavailable."""


__author__ = ["Ankit-1204"]


class TabDataset(DataSet):
    """Implements Pytorch Dataset class for Training TabTransformer."""

    def __init__(self, X, y, cat_idx) -> None:
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)

        self.cat_idx = cat_idx
        self._prepare_data()

    def _prepare_data(self):
        self.x_cat = self.X[:, self.cat_idx].long()
        self.x_con = self.X[
            :, [i for i in range(self.X.shape[1]) if i not in self.cat_idx]
        ]

    def __len__(self):
        """Get length of the dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Return data point."""
        return self.x_cat[index], self.x_con[index], self.y[index]


class Tab_Transformer(NNModule):
    r"""Tab Transformer Architecture.

    Parameters
    ----------
    num_cat_class : list,
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

    """

    def __init__(
        self,
        num_cat_class,
        num_cont_features,
        embedding_dim,
        n_transformer_layer,
        n_heads,
        output_dim,
    ) -> None:
        super().__init__()
        self.num_cat_class = num_cat_class
        self.num_cont_features = num_cont_features
        self.embedding_dim = embedding_dim
        self.n_transformer_layer = n_transformer_layer
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.embedding = nn.ModuleList(
            [nn.Embedding(cat, self.embedding_dim) for cat in self.num_cat_class]
        )
        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=self.n_heads, batch_first=True
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, num_layers=self.n_transformer_layer
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                self.embedding_dim * len(self.num_cat_class) + self.num_cont_features,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, x_cat, x_cont):
        """Implement forward for Tab Transformer."""
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding)]
        embedded = torch.stack(dim=1)

        context = self.transformer(embedded)
        context = context.view(context.size(0), -1)
        combined = torch.concat([context, x_cont], dim=1)
        feed = self.feed_forward(combined)
        return feed


class TabTransformerRegressor(BaseDeepNetworkPyTorch):
    r"""
    Tab Transformer for Tabular Data.

    Parameters
    ----------
    num_cat_class : list,
        array of unique classes for each categorical feature
    num_cont_features : int,
        number of continuos features
    embedding dim : int,
        value of embedding dimension
    n_transformer_layer : int,
        number of transformer layers
    n_heads : int,
        number of attention heads
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
        num_cat_class,
        cat_idx,
        num_cont_features,
        embedding_dim,
        n_transformer_layer,
        n_heads,
        num_epochs=16,
        batch_size=8,
        lr=0.001,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
    ):
        self.num_cat_class = num_cat_class
        self.cat_idx = cat_idx
        self.num_cont_features = num_cont_features
        self.embedding_dim = embedding_dim
        self.n_transformer_layer = n_transformer_layer
        self.n_heads = n_heads
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.lr = lr
        super().__init__()
        if _check_soft_dependencies("torch", severity="none"):
            import torch

            self.criterions = {
                "MSE": torch.nn.MSELoss,
                "L1": torch.nn.L1Loss,
                "SmoothL1": torch.nn.SmoothL1Loss,
                "Huber": torch.nn.HuberLoss,
            }

            self.optimizers = {
                "Adadelta": torch.optim.Adadelta,
                "Adagrad": torch.optim.Adagrad,
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "SGD": torch.optim.SGD,
            }

    def _build_network(self, y):
        self.output_dim = y.shape[0]
        self.network = Tab_Transformer(
            self.num_cat_class,
            self.num_cont_features,
            self.embedding_dim,
            self.n_transformer_layer,
            self.n_heads,
            self.output_dim,
        )

    def build_pytorch_train_dataloader(self, X, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_train:
            if hasattr(self.custom_dataset_train, "build_dataset") and callable(
                self.custom_dataset_train.build_dataset
            ):
                self.custom_dataset_train.build_dataset(y)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please "
                    f"refer to the {self.__class__.__name__}.build_dataset "
                    "documentation."
                )
        else:
            dataset = TabDataset(X=X, y=y, cat_idx=self.cat_idx)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def _fit(self, X, y):
        r"""Fit the model on input.

        Parameters
        ----------
        X :
        y : 1d or 2d numpy array
            for classification, shape = (n_instances)
            for regression, shape = (n_instances, n_dimensions)
        """
        dataloader = self._build_network(X, y)
        dataloader
        return
