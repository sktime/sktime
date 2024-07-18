"""Pytorch Transformer Model."""

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch


class TransformerClassifier(BaseDeepClassifierPytorch):
    """Transformer Classifier based on Pytorch."""

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
    }

    def __init__(
        self,
        # model specific
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        num_classes,
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
    ):
        self.feat_dim = feat_dim
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
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

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            verbose=verbose,
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

    def _build_network(self):
        from sktime.classification.deep_learning.transformer.network import (
            TSTransformerEncoderClassiregressor,
        )

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
        params = []
        return params
