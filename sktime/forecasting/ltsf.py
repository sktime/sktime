"""Deep Learning Forecasters using LTSF-Linear Models."""

import pandas as pd

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.ltsf.data.dataset import PytorchFormerDataset


class LTSFLinearForecaster(BaseDeepNetworkPyTorch):
    """LTSF-Linear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) linear forecaster,
    aka LTSF-Linear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    num_epochs : int, default=16
        number of epochs to train
    batch_size : int, default=8
        number of training examples per batch
    in_channels : int, default=1
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    criterion_kwargs : dict, default=None
        keyword arguments to pass to criterion
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    optimizer_kwargs : dict, default=None
        keyword arguments to pass to optimizer
    lr : float, default=0.003
        learning rate to train model with

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear

    Examples
    --------
    >>> from sktime.forecasting.ltsf import LTSFLinearForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> model = LTSFLinearForecaster(10, 3) # doctest: +SKIP
    >>> y = load_airline()
    >>> model.fit(y, fh=[1,2,3]) # doctest: +SKIP
    LTSFLinearForecaster(pred_len=3, seq_len=10)
    >>> y_pred = model.predict() # doctest: +SKIP
    >>> y_pred # doctest: +SKIP
    1961-01    515.456726
    1961-02    576.704712
    1961-03    559.859680
    Freq: M, Name: Number of airline passengers, dtype: float32
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["luca-miniati"],
        "maintainers": ["luca-miniati"],
        # "python_dependencies": "pytorch" - inherited from BaseDeepNetworkPyTorch
        # estimator type vars inherited from BaseDeepNetworkPyTorch
    }

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.in_channels = in_channels
        self.criterion = criterion
        self.optimizer = optimizer
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.num_epochs = num_epochs
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred
        self.batch_size = batch_size

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
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
        from sktime.networks.ltsf.models.linear import LTSFLinearNetwork

        return LTSFLinearNetwork(
            self.seq_len,
            fh,
            self.in_channels,
            self.individual,
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
                "seq_len": 2,
                "pred_len": 1,
                "lr": 0.005,
                "optimizer": "Adam",
                "batch_size": 1,
                "num_epochs": 1,
                "individual": True,
            }
        ]

        return params


class LTSFDLinearForecaster(BaseDeepNetworkPyTorch):
    """LTSF-DLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) decomposition linear
    forecaster, aka LTSF-DLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    num_epochs : int, default=16
        number of epochs to train
    batch_size : int, default=8
        number of training examples per batch
    in_channels : int, default=1
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    criterion_kwargs : dict, default=None
        keyword arguments to pass to criterion
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    optimizer_kwargs : dict, default=None
        keyword arguments to pass to optimizer
    lr : float, default=0.003
        learning rate to train model with

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear

    Examples
    --------
    >>> from sktime.forecasting.ltsf import LTSFDLinearForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> model = LTSFDLinearForecaster(10, 3) # doctest: +SKIP
    >>> y = load_airline()
    >>> model.fit(y, fh=[1,2,3]) # doctest: +SKIP
    LTSFDLinearForecaster(pred_len=3, seq_len=10)
    >>> y_pred = model.predict() # doctest: +SKIP
    >>> y_pred # doctest: +SKIP
    1961-01    436.494476
    1961-02    433.659851
    1961-03    479.309631
    Freq: M, Name: Number of airline passengers, dtype: float32
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.in_channels = in_channels
        self.criterion = criterion
        self.optimizer = optimizer
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.num_epochs = num_epochs
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred
        self.batch_size = batch_size

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
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
        from sktime.networks.ltsf.models.linear import LTSFDLinearNetwork

        return LTSFDLinearNetwork(
            self.seq_len,
            fh,
            self.in_channels,
            self.individual,
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
                "seq_len": 2,
                "pred_len": 1,
                "lr": 0.005,
                "optimizer": "Adam",
                "batch_size": 1,
                "num_epochs": 1,
                "individual": True,
            }
        ]

        return params


class LTSFNLinearForecaster(BaseDeepNetworkPyTorch):
    """LTSF-NLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) normalization linear
    forecaster, aka LTSF-NLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    num_epochs : int, default=16
        number of epochs to train
    batch_size : int, default=8
        number of training examples per batch
    in_channels : int, default=1
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    criterion_kwargs : dict, default=None
        keyword arguments to pass to criterion
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    optimizer_kwargs : dict, default=None
        keyword arguments to pass to optimizer
    lr : float, default=0.003
        learning rate to train model with

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear

    Examples
    --------
    >>> from sktime.forecasting.ltsf import LTSFNLinearForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> model = LTSFNLinearForecaster(10, 3) # doctest: +SKIP
    >>> y = load_airline()
    >>> model.fit(y, fh=[1,2,3]) # doctest: +SKIP
    LTSFNLinearForecaster(pred_len=3, seq_len=10)
    >>> y_pred = model.predict() # doctest: +SKIP
    >>> y_pred # doctest: +SKIP
    1961-01    455.628082
    1961-02    433.349640
    1961-03    437.045502
    Freq: M, Name: Number of airline passengers, dtype: float32
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.in_channels = in_channels
        self.criterion = criterion
        self.optimizer = optimizer
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.num_epochs = num_epochs
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred
        self.batch_size = batch_size

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
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
        from sktime.networks.ltsf.models.linear import LTSFNLinearNetwork

        return LTSFNLinearNetwork(
            self.seq_len,
            fh,
            self.in_channels,
            self.individual,
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
                "seq_len": 2,
                "pred_len": 1,
                "lr": 0.005,
                "optimizer": "Adam",
                "batch_size": 1,
                "num_epochs": 1,
                "individual": True,
            }
        ]

        return params


class LTSFTransfomer(BaseDeepNetworkPyTorch):
    """LTSF-Transformer Forecaster.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
    pred_len : int
        Length of the prediction sequence.
    context_len : int, optional (default=2)
        Length of the label sequence.
    num_epochs : int, optional (default=16)
        Number of epochs for training.
    batch_size : int, optional (default=8)
        Size of the batch.
    in_channels : int, optional (default=1)
        Number of input channels.
    individual : bool, optional (default=False)
        Whether to use individual models for each series.
    criterion : str or callable, optional
        Loss function to use.
    criterion_kwargs : dict, optional
        Additional keyword arguments for the loss function.
    optimizer : str or callable, optional
        Optimizer to use.
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float, optional (default=0.001)
        Learning rate.
    custom_dataset_train : torch.utils.data.Dataset, optional
        Custom dataset for training.
    custom_dataset_pred : torch.utils.data.Dataset, optional
        Custom dataset for prediction.
    output_attention : bool, optional (default=False)
        Whether to output attention weights.
    embed_type : int, optional (default=0)
        Type of embedding to use.
    embed : str, optional (default="fixed")
        Type of embedding.
    enc_in : int, optional (default=7)
        Number of encoder input features.
    dec_in : int, optional (default=7)
        Number of decoder input features.
    d_model : int, optional (default=512)
        Dimension of the model.
    n_heads : int, optional (default=8)
        Number of attention heads.
    d_ff : int, optional (default=2048)
        Dimension of the feed-forward network.
    e_layers : int, optional (default=3)
        Number of encoder layers.
    d_layers : int, optional (default=2)
        Number of decoder layers.
    factor : int, optional (default=5)
        Factor for attention.
    dropout : float, optional (default=0.1)
        Dropout rate.
    activation : str, optional (default="relu")
        Activation function.
    c_out : int, optional (default=7)
        Number of output features.
    freq : str, optional (default="h")
        Frequency of the data.

    Examples
    --------
    >>> from sktime.forecasting.ltsf import LTSFTransfomer, LTSFLinearForecaster
    >>> from sktime.datasets import load_longley
    >>>
    >>> batch_size = 5
    >>> seq_len = 5
    >>> context_len = 2
    >>> pred_len = 3
    >>> num_features = 1
    >>>
    >>> y, X = load_longley()
    >>> split_point = len(y) - pred_len
    >>> X_train, X_test = X[:split_point], X[split_point:]
    >>> y_train, y_test = y[:split_point], y[split_point:]
    >>>
    >>> model = LTSFTransfomer(
    ... 	seq_len = seq_len,
    ... 	pred_len = pred_len,
    ... 	context_len = context_len,
    ... 	output_attention = False,
    ... 	embed_type = 0,
    ... 	embed = "fiixed",
    ... 	enc_in = num_features,
    ... 	dec_in = num_features,
    ... 	d_model = 512,
    ... 	n_heads = 8,
    ... 	d_ff = 2048,
    ... 	e_layers = 1,
    ... 	d_layers = 1,
    ... 	factor = 5,
    ... 	dropout = 0.1,
    ... 	activation = "relu",
    ... 	c_out = pred_len,
    ... 	freq = 'h',
    ... 	num_epochs=1,
    ... 	batch_size=batch_size,
    >>> )
    >>>
    >>> model.fit(y_train, X_train, fh=[1, 2, 3])
    >>> pred = model.predict(X=X_test)



    """

    def __init__(
        self,
        seq_len,
        context_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,

        position_encoding: bool = True,
        temporal_encoding: bool = True,
        temporal_encoding_type = "linear", # linear, embed, fixed-embed

        d_model=512,
        n_heads=8,
        d_ff=2048,
        e_layers=3,
        d_layers=2,
        factor=5,
        dropout=0.1,
        activation="relu",
        freq="h",
    ):

        """
        suggested in the paper
        context_len = pred_len
        seq_len = 2 * pred_len

        """

        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len

        self.individual = individual
        self.in_channels = in_channels
        self.criterion = criterion
        self.optimizer = optimizer
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.num_epochs = num_epochs
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred
        self.batch_size = batch_size

        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.temporal_encoding_type = temporal_encoding_type

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.freq = freq

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
        )

        self.output_attention = False # attention in output is not needed by the user

        if self.temporal_encoding:
            from sktime.networks.ltsf.utils.timefeatures import get_mark_vocab_sizes
            self.mark_vocab_sizes = get_mark_vocab_sizes(
                temporal_encoding_type=self.temporal_encoding_type,
                freq=self.freq,
            )
        else:
            self.mark_vocab_sizes = None

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

    def build_pytorch_train_dataloader(self, y):
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
            dataset = PytorchFormerDataset(
                y=y,
                seq_len=self.seq_len,
                context_len=self.context_len,
                pred_len=self.pred_len,
                freq=self.freq,
                temporal_encoding=self.temporal_encoding,
                temporal_encoding_type=self.temporal_encoding_type,
            )

        return DataLoader(dataset, self.batch_size, shuffle=True)

    def build_pytorch_pred_dataloader(self, y, fh):
        """Build PyTorch DataLoader for prediction."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_pred:
            if hasattr(self.custom_dataset_pred, "build_dataset") and callable(
                self.custom_dataset_pred.build_dataset
            ):
                self.custom_dataset_train.build_dataset(y)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please"
                    f"refer to the {self.__class__.__name__}.build_dataset"
                    "documentation."
                )
        else:
            mask_y = pd.DataFrame(
                data=0, columns=y.columns, index=fh.to_absolute_index(self.cutoff)
            )
            _y = y.iloc[-self.seq_len :]
            _y = pd.concat([_y, mask_y], axis=0)

            dataset = PytorchFormerDataset(
                y=_y,
                seq_len=self.seq_len,
                context_len=self.context_len,
                pred_len=self.pred_len,
                freq=self.freq,
                temporal_encoding=self.temporal_encoding,
                temporal_encoding_type=self.temporal_encoding_type,
            )

        return DataLoader(
            dataset,
            self.batch_size,
        )

    def _build_network(self, fh):
        from sktime.networks.ltsf.models.transformers import LTSFTransformerNetwork

        num_features = self._y.shape[-1]

        self.enc_in = num_features
        self.dec_in = num_features
        self.c_out = num_features

        class Configs:
            def __init__(self_config):
                self_config.seq_len = self.seq_len
                self_config.context_len = self.context_len
                self_config.pred_len = self.pred_len
                self_config.output_attention = self.output_attention
                self_config.mark_vocab_sizes = self.mark_vocab_sizes
                self_config.position_encoding = self.position_encoding
                self_config.temporal_encoding = self.temporal_encoding
                self_config.temporal_encoding_type = self.temporal_encoding_type
                self_config.enc_in = self.enc_in
                self_config.dec_in = self.dec_in
                self_config.d_model = self.d_model
                self_config.n_heads = self.n_heads
                self_config.d_ff = self.d_ff
                self_config.e_layers = self.e_layers
                self_config.d_layers = self.d_layers
                self_config.factor = self.factor
                self_config.dropout = self.dropout
                self_config.activation = self.activation
                self_config.c_out = self.c_out

        return LTSFTransformerNetwork(Configs())._build()


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
                "seq_len": 2,
                "pred_len": 1,
                "lr": 0.005,
                "optimizer": "Adam",
                "batch_size": 1,
                "num_epochs": 1,
                "individual": True,
            }
        ]

        return params
