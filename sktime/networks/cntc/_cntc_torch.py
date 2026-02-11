"""CNTC for Classification and Regression in PyTorch."""

__authors__ = ["fnhirwa"]
__all__ = ["CNTCNetworkTorch"]


from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class CNTCNetworkTorch(NNModule):
    """CNTC (Contextual Neural Networks for Time Series Classification).

    CNTC is a deep learning network architecture for time series classification
    and regression. It combines convolutional layers, recurrent layers,
    attention mechanisms, and dense layers to capture both local
    and global patterns in time series data.

    For more details on the architecture, see [1]_

    Parameters
    ----------
    kernel_size: int, default = 7
        The size of the 1D convolutional kernel.
    evg_pool_size: int, default = 3
        The size of the pooling windows for the evg pooling layer.
    n_conv_layers: int, default = 2
        The number of convolutional layers in the network (plus the evg pooling layer).
    filter_sizes: list of int, with the shape (n_conv_layers,), default = [64, 128]
        The number of filters in each convolutional layer.
    activation: str, default = 'relu'
        The activation function used inside hidden layers
        (excluding the self attention module).
        List of available PyTorch activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations
    activation_attention: str, default = 'sigmoid'
        The activation function used inside the self attention module.
        List of available PyTorch activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations
    dropout: float or tuple of floats, default = (0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1)
        The dropout rate(s) used in the network, in the range [0, 1).
        If a single float is provided, the same dropout rate is applied to all layers.
        If a tuple is provided, it should have 7 values corresponding to:
        (conv1_dropout, rnn1_dropout, conv2_dropout, lstm_dropout,
         avg_dropout, att_dropout, mlp_dropout)
        where mlp_dropout is applied to both MLP layers.
    random_state: int, default = 0
        Seed for any needed random actions.

    References
    ----------
    .. [1] Network originally defined in:
        @article{FULLAHKAMARA202057,
        title = {Combining contextual neural networks for time series classification},
        journal = {Neurocomputing},
        volume = {384},
        pages = {57-66},
        year = {2020},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2019.10.113},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231219316364},
        author = {Amadu {Fullah Kamara} and Enhong Chen and Qi Liu and Zhen Pan},
        keywords = {Time series classification, Contextual convolutional neural
            networks, Contextual long short-term memory, Attention, Multilayer
            perceptron},
       }
    """

    _tags = {
        # packaging info
        # --------------
        "authors": __authors__,
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        kernel_size: int = 7,
        evg_pool_size: int = 3,
        n_conv_layers: int = 2,
        filter_sizes: list[int] = [64, 128],
        activation: str = "relu",
        activation_attention: str = "sigmoid",
        dropout: float | tuple[float, float, float, float, float, float, float] = (
            0.2,
            0.2,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ),
        random_state: int = 0,
    ):
        pass
