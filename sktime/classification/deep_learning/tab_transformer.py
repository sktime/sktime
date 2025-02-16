"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

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


class Tab_Transformer(NNModule):
    """Tab Transformer Architecture."""

    def __init__(
        self,
        num_cat_feat,
        num_cont_features,
        embedding_dim,
        n_transformer_layer,
        n_heads,
    ) -> None:
        super().__init__()
        self.embedding = nn.ModuleList(
            [nn.Embedding(cat, embedding_dim) for cat in num_cat_feat]
        )
        self.transformer_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, num_layers=n_transformer_layer
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim * num_cat_feat + num_cont_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat, x_cont):
        """Implement forward for Tab Transformer."""
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding)]
        embedded = torch.stack(dim=1)

        context = self.transformer(embedded)
        context


class TabTransformer(BaseDeepNetworkPyTorch):
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
