"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.datatypes._convert import convert_to
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""


class Tab_Transformer(NNModule):
    """Tab Transformer Architecture."""

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
                self.embedding_dim * self.num_cat_feat + self.num_cont_features, 128
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


class TabTransformer(BaseDeepClassifierPytorch):
    r"""
    Tab Transformer for Tabular Data.

    References
    ----------
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
    Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
    https://arxiv.org/pdf/2012.06678
    """

    def __init__(
        self,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
    ):
        super().__init__(
            num_epochs,
            batch_size,
            in_channels,
            individual,
            criterion_kwargs,
            optimizer,
            optimizer_kwargs,
            lr,
        )
        if _check_soft_dependencies("torch", severity="none"):
            import torch.nn as nn

            NNModule = nn.Module
        else:

            class NNModule:
                """Dummy class if torch is unavailable."""

    def _window(self, X, y, window_size=3):
        r"""Convert Timeseries into Tabular Format.

        Parameters
        ----------
        Window Size : int, Optional (default=3)
        Defines the size of the sliding window

        X : Nested Dataframe
        y : Optional (Required for Classification)
        """
        X_new = convert_to(X, to_type="numpyflat", as_scitype="Panel")
        X_new
        return
