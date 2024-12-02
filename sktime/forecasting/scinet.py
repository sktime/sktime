"""Deep Learning Forecaster using SCINet Forecaster."""

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch


class SCINetForecaster(BaseDeepNetworkPyTorch):
    """SCINet Forecaster.

    Implementation of the SCINet forecaster,  by Minhao Liu* [1]_.

    Core logic is directly copied from the curelab SCINet implementation
    [2]_, which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
        Ensure seq_len is divisible by 2^num_levels.

    num_epochs : int, default=16
        Number of epochs to train the model.

    batch_size : int, default=8
        Number of training examples in each batch.

    criterion : torch.nn Loss Function, default=None
        Loss function to be used for training. If not provided, a default such as
        torch.nn.MSELoss is often used.

    criterion_kwargs : dict, default=None
        Keyword arguments to pass to the criterion (loss function).

    optimizer : torch.optim.Optimizer, default=None
        Optimizer to be used for training. If not provided, a default such as
        torch.optim.Adam is commonly used.

    optimizer_kwargs : dict, default=None
        Keyword arguments to pass to the optimizer.

    lr : float, default=0.001
        Learning rate for the optimizer.

    custom_dataset_train : Dataset, default=None
        A custom dataset to be used for training. If not provided, the default dataset
        structure is used.

    custom_dataset_pred : Dataset, default=None
        A custom dataset to be used for prediction.

    hid_size : int, default=1
        Size of the hidden layers in the model.

    num_stacks : int, default=1
        Number of SCINet stacks to use in the model.

    num_levels : int, default=3
        Number of levels (depth) in each stack.

    num_decoder_layer : int, default=1
        Number of layers in the decoder portion of the model.

    concat_len : int, default=0
        Length of input to be concatenated in the skip connection.

    groups : int, default=1
        Number of groups in convolution layers for grouped convolutions.

    kernel : int, default=5
        Kernel size for convolution layers.

    dropout : float, default=0.5
        Dropout rate to apply in the network.

    single_step_output_One : int, default=0
        Determines whether to output a single step (1) or multiple steps (0).

    positionalE : bool, default=False
        Enables or disables the use of positional encoding.

    modified : bool, default=True
        Indicates whether to use the modified version of the SCINet model.

    RIN : bool, default=False
        Flag to enable or disable the use of RevIN (Reversible Instance Normalization).

    Raises
    ------
    AssertionError
        If seq_len is not divisible by 2^num_levels.

    References
    ----------
    .. [1] Minhao Liu*, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai,
    Lingna Ma, Qiang Xu*
    SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction

    .. [2] https://github.com/cure-lab/SCINet

    Examples
    --------
    >>> from sktime.forecasting.scinet import SCINetForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline # doctest: +SKIP
    >>> model = SCINetForecaster(seq_len=8) # doctest: +SKIP
    >>> y = load_airline() # doctest: +SKIP
    >>> model.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    SCINetForecaster(seq_len=8)
    >>> y_pred = model.predict() # doctest: +SKIP
    >>> y_pred # doctest: +SKIP
    1961-01    759.448425
    1961-02    291.098541
    1961-03    566.977295
    Freq: M, Name: Number of airline passengers, dtype: float32
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "ailingzengzzz",
            "AlexMinhao",
            "VEWOXIC",
            "mixiancmx",
            "Sohaib-Ahmed21",
        ],
        # mixiancmx, ailingzengzzz, AlexMinhao, VEWOXIC for cure-lab code
        "maintainers": ["Sohaib-Ahmed21"],
        # "python_dependencies": "pytorch" - inherited from BaseDeepNetworkPyTorch
        # estimator type vars inherited from BaseDeepNetworkPyTorch
    }

    def __init__(
        self,
        seq_len,
        *,
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
        hid_size=1,
        num_stacks=1,
        num_levels=3,
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=5,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=False,
        modified=True,
        RIN=False,
    ):
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizer = optimizer
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.num_epochs = num_epochs
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred
        self.batch_size = batch_size
        self.hid_size = hid_size
        self.num_stacks = num_stacks
        self.num_levels = num_levels
        self.num_decoder_layer = num_decoder_layer
        self.concat_len = concat_len
        self.groups = groups
        self.kernel = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.positionalE = positionalE
        self.modified = modified
        self.RIN = RIN

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
        )

        from sktime.utils.dependencies import _check_soft_dependencies

        if _check_soft_dependencies("torch"):
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

    def _build_network(self, fh):
        # Define the SCINet-based network
        from sktime.networks.scinet import SCINet

        return SCINet(
            seq_len=self.seq_len,
            input_dim=self._y.shape[-1],
            pred_len=fh,
            hid_size=self.hid_size,
            num_stacks=self.hid_size,
            num_levels=self.num_levels,
            num_decoder_layer=self.num_decoder_layer,
            concat_len=self.concat_len,
            groups=self.groups,
            kernel=self.kernel,
            dropout=self.dropout,
            single_step_output_One=self.single_step_output_One,
            positionalE=self.positionalE,
            modified=self.modified,
            RIN=self.RIN,
        )._build()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        params = [
            {
                "seq_len": 8,
                "lr": 0.005,
                "optimizer": "Adam",
                "batch_size": 1,
                "num_epochs": 1,
            },
            {
                "seq_len": 16,
                "lr": 0.001,
                "optimizer": "Adam",
                "batch_size": 4,
                "num_epochs": 2,
            },
        ]

        return params
