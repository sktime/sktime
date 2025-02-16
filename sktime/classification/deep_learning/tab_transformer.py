"""TabTransformer : Modeling Tabular Data using Contextual Embedding."""

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""


class Tab_Transformer(NNModule):
    """Tab Transformer Architecture."""

    def __init__(self, num_cat_feat, embedding_dim, num_layer) -> None:
        super().__init__()


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
