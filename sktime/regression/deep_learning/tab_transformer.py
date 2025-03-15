"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

from sktime.regression.base import BaseRegressor
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    DataSet = Dataset
    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""

    class DataSet:
        """Dummy class if torch is unavailable."""


class TabTrainDataset(DataSet):
    """Implements Pytorch Dataset class for Training TabTransformer."""

    def __init__(self, X, y, cat_idx) -> None:
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)
        self.cat_idx = cat_idx
        self._prepare_data()

    def _prepare_data(self):
        self.x_cat = self.X[:, :, self.cat_idx].long()
        self.x_con = self.X[
            :, :, [i for i in range(self.X.shape[2]) if i not in self.cat_idx]
        ]

    def __len__(self):
        """Get length of the dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Return data point."""
        return self.x_cat[index], self.x_con[index].float(), self.y[index].float()


class TabPredDataset(DataSet):
    """Implements Pytorch Dataset class for Predicting with TabTransformer."""

    def __init__(self, X, cat_idx) -> None:
        self.X = torch.tensor(X)
        self.cat_idx = cat_idx
        self._prepare_data()

    def _prepare_data(self):
        self.x_cat = self.X[:, :, self.cat_idx].long()
        self.x_con = self.X[
            :, :, [i for i in range(self.X.shape[2]) if i not in self.cat_idx]
        ]

    def __len__(self):
        """Get length of the dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Return data point."""
        return self.x_cat[index], self.x_con[index].float()


class Tab_Transformer(NNModule):
    r"""Tab Transformer Architecture.

    Parameters
    ----------
    num_cat_class : list,
        array of number unique classes for each categorical feature
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
        if _check_soft_dependencies("torch", severity="none"):
            import torch.nn as nn
            from torch.nn import TransformerEncoder, TransformerEncoderLayer

            self.embedding = nn.ModuleList(
                [nn.Embedding(cat, self.embedding_dim) for cat in self.num_cat_class]
            )
            self.cat_embed = max(self.embedding_dim * len(self.num_cat_class), 2)
            self.transformer_layer = TransformerEncoderLayer(
                d_model=self.cat_embed,
                nhead=self.n_heads,
                batch_first=True,
            )
            self.transformer = TransformerEncoder(
                self.transformer_layer, num_layers=self.n_transformer_layer
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(
                    self.embedding_dim * len(self.num_cat_class)
                    + self.num_cont_features,
                    128,
                ),
                nn.ReLU(),
                nn.Linear(128, self.output_dim),
            )

    def forward(self, x_cat, x_cont):
        """Implement forward for Tab Transformer."""
        if x_cat.shape[2] == 0:
            combined = x_cont
        else:
            embedded = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embedding)]
            embedded = torch.cat(embedded, dim=2)
            context = self.transformer(embedded)
            combined = torch.concat([context, x_cont], dim=2)

        feed = self.feed_forward(combined)
        return feed[:, -1, :]


class TabTransformerRegressor(BaseRegressor):
    r"""
    Tab Transformer for Tabular Data.

    Parameters
    ----------
    num_cat_class : list,
        array of unique classes for each categorical feature
    cat_idx : list,
        Indices for the categorcal features
    embedding dim : int,
        value of embedding dimension
    n_transformer_layer : int, default = 6
        number of transformer layers
    n_heads : int,
        number of attention heads
    num_epochs : int, default = 100
        the number of epochs to train the model
    batch_size : int, default = 8
        the number of samples per gradient update.
    lr : int
        Learning rate for training
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    criterion_kwargs : dict, default=None
        keyword arguments to pass to criterion
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    optimizer_kwargs : dict, default=None
        keyword arguments to pass to optimizer

    References
    ----------
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
    Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
    https://arxiv.org/pdf/2012.06678

    Examples
    --------
    >>> from sktime.regression.deep_learning import TabTransformerRegressor
    >>> from sktime.datasets import load_unit_test
    >>> X_train, Y_train = load_unit_test(split="train")
    >>> clf =TabTransformerRegressor(num_epochs=50, batch_size=4)# doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ankit-1204"],
        "python_dependencies": "torch",
        # estimator type
        # --------------
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        num_cat_class=None,
        cat_idx=None,
        embedding_dim=6,
        n_transformer_layer=4,
        n_heads=1,
        num_epochs=100,
        batch_size=8,
        lr=0.01,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        custom_dataset_pred=None,
        custom_dataset_train=None,
    ):
        self.num_cat_class = num_cat_class
        self.cat_idx = cat_idx
        self.embedding_dim = embedding_dim
        self.n_transformer_layer = n_transformer_layer
        self.n_heads = n_heads
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.custom_dataset_pred = custom_dataset_pred
        self.custom_dataset_train = custom_dataset_train
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

    def _build_network(self, X, y):
        if y.ndim == 1:
            self.output_dim = 1
        else:
            self.output_dim = y.shape[1]
        if self.num_cat_class is None:
            self.num_cat_class = []
        if self.cat_idx is None:
            self.cat_idx = []
        self.num_cont_features = X.shape[2] - len(self.cat_idx)
        return Tab_Transformer(
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
                self.custom_dataset_train.build_dataset(X, self.cat_idx)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please "
                    f"refer to the {self.__class__.__name__}.build_dataset "
                    "documentation."
                )
        else:
            dataset = TabTrainDataset(X=X, y=y, cat_idx=self.cat_idx)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def build_pytorch_pred_dataloader(self, X):
        """Build PyTorch DataLoader for prediction."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_pred:
            if hasattr(self.custom_dataset_pred, "build_dataset") and callable(
                self.custom_dataset_pred.build_dataset
            ):
                self.custom_dataset_train.build_dataset(X, self.cat_idx)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please"
                    f"refer to the {self.__class__.__name__}.build_dataset"
                    "documentation."
                )
        else:
            dataset = TabPredDataset(X=X, cat_idx=self.cat_idx)

        return DataLoader(
            dataset,
            self.batch_size,
        )

    def _instantiate_optimizer(self):
        import torch

        if self.optimizer:
            if self.optimizer in self.optimizers.keys():
                if self.optimizer_kwargs:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr
                    )
            else:
                raise TypeError(
                    f"Please pass one of {self.optimizers.keys()} for `optimizer`."
                )
        else:
            # default optimizer
            return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _instantiate_criterion(self):
        import torch

        if self.criterion:
            if self.criterion in self.criterions.keys():
                if self.criterion_kwargs:
                    return self.criterions[self.criterion](**self.criterion_kwargs)
                else:
                    return self.criterions[self.criterion]()
            else:
                raise TypeError(
                    f"Please pass one of {self.criterions.keys()} for `criterion`."
                )
        else:
            # default criterion
            return torch.nn.MSELoss()

    def _run_epoch(self, epoch, dataloader):
        for x_cat, x_con, y in dataloader:
            y_pred = self.network(x_cat, x_con)
            loss = self._criterion(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _fit(self, X, y):
        r"""Fit the model on input.

        Parameters
        ----------
        X : 3d numpy array, shape = (n_instances,n_features,seq_length)
        y : 1d or 2d numpy array
            for regression, shape = (n_instances, n_dimensions)
        """
        X = X.transpose(0, 2, 1)
        self.network = self._build_network(X, y)
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        dataloader = self.build_pytorch_train_dataloader(X, y)
        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _predict(self, X):
        r"""Predict with fitted model.

        Parameters
        ----------
        X : 3d numpy array, shape = (n_instances,n_features,seq_length)
        """
        from torch import cat

        X = X.transpose(0, 2, 1)
        dataloader = self.build_pytorch_pred_dataloader(X)
        self.network.eval()
        y_pred = []
        for x_cat, x_cont in dataloader:
            y_pred.append(self.network(x_cat, x_cont).detach())
        y_pred = cat(y_pred, dim=0).view(-1, y_pred[0].shape[-1]).numpy()
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str , default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params :dict or list of dict , default = {}
            parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in `params
        """
        params = [
            {},
            {
                "num_cat_class": None,
                "cat_idx": None,
                "embedding_dim": 8,
                "n_transformer_layer": 3,
                "n_heads": 2,
                "num_epochs": 50,
                "batch_size": 16,
                "lr": 0.1,
                "criterion": None,
                "criterion_kwargs": None,
                "optimizer": None,
                "optimizer_kwargs": None,
            },
        ]

        return params
