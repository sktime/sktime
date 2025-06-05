"""Deep Learning Forecasters using LTSF-Linear Models."""

import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.warnings import warn


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
        "authors": ["mixiancmx", "ailingzengzzz", "luca-miniati"],
        # mixiancmx, ailingzengzzz for cure-lab code
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

    _tags = {
        # packaging info
        # --------------
        "authors": ["mixiancmx", "ailingzengzzz", "luca-miniati"],
        # mixiancmx, ailingzengzzz for cure-lab code
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

    _tags = {
        # packaging info
        # --------------
        "authors": ["mixiancmx", "ailingzengzzz", "luca-miniati"],
        # mixiancmx, ailingzengzzz for cure-lab code
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


class LTSFTransformerForecaster(BaseDeepNetworkPyTorch):
    """LTSF-Transformer Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) transformer forecaster,
    aka LTSF-Transformer, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
        Preferred to be twice the pred_len.
    pred_len : int
        Length of the prediction sequence.
    context_len : int, optional (default=2)
        Length of the label sequence.
        Preferred to be same as the pred_len.
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
    position_encoding : bool, optional (default=True)
        Whether to use positional encoding.
        Positional encoding helps the model understand the order of elements
        in the input sequence by adding unique positional information to each element.
    temporal_encoding : bool, optional (default=True)
        Whether to use temporal encoding.
        Works only with DatetimeIndex and PeriodIndex, disabled otherwise.
    temporal_encoding_type : str, optional (default="linear")
        Type of temporal encoding to use, relevant only if temporal_encoding is True.
        - "linear": Uses linear layer to encode temporal data.
        - "embed": Uses embeddings layer with learnable weights.
        - "fixed-embed": Uses embeddings layer with fixed sine-cosine values as weights.
    d_model : int, optional (default=512)
        Dimension of the model.
    n_heads : int, optional (default=8)
        Number of attention heads.
    d_ff : int, optional (default=2048)
        Dimension of the feedforward network model.
    e_layers : int, optional (default=3)
        Number of encoder layers.
    d_layers : int, optional (default=2)
        Number of decoder layers.
    factor : int, optional (default=5)
        Factor for the attention mechanism.
    dropout : float, optional (default=0.1)
        Dropout rate.
    activation : str, optional (default="relu")
        Activation function to use. Defaults to relu and otherwise gelu.
    freq : str, optional (default="h")
        Frequency of the input data, relevant only if temporal_encoding is True.

    Examples
    --------
    >>> from sktime.forecasting.ltsf import LTSFTransformerForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>>
    >>> y = load_airline()
    >>>
    >>> model = LTSFTransformerForecaster(10, 5, 5) # doctest: +SKIP
    >>> model.fit(y, fh=[1, 2, 3, 4, 5]) # doctest: +SKIP
    LTSFTransformerForecaster(context_len=5, pred_len=5, seq_len=10)
    >>> pred = model.predict() # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["mixiancmx", "ailingzengzzz", "geetu040"],
        # mixiancmx, ailingzengzzz for cure-lab code
        "maintainers": ["geetu040"],
        # "python_dependencies": "pytorch" - inherited from BaseDeepNetworkPyTorch
        # estimator type vars inherited from BaseDeepNetworkPyTorch
    }

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
        position_encoding=True,
        temporal_encoding=True,
        temporal_encoding_type="linear",  # linear, embed, fixed-embed
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
        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self._pred_len = None

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
        self._temporal_encoding = None
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
            from sktime.networks.ltsf.data.dataset import PytorchFormerDataset

            dataset = PytorchFormerDataset(
                y=y,
                seq_len=self.seq_len,
                context_len=self.context_len,
                pred_len=self._pred_len,
                freq=self.freq,
                temporal_encoding=self._temporal_encoding,
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
            _fh = ForecastingHorizon(range(1, self._pred_len + 1), is_relative=True)
            mask_y = pd.DataFrame(
                data=0, columns=y.columns, index=_fh.to_absolute_index(self.cutoff)
            )
            _y = y.iloc[-self.seq_len :]
            _y = pd.concat([_y, mask_y], axis=0)

            from sktime.networks.ltsf.data.dataset import PytorchFormerDataset

            dataset = PytorchFormerDataset(
                y=_y,
                seq_len=self.seq_len,
                context_len=self.context_len,
                pred_len=self._pred_len,
                freq=self.freq,
                temporal_encoding=self._temporal_encoding,
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

        self.output_attention = False  # attention in output is not needed by the user

        self._pred_len = fh

        if self.temporal_encoding:
            if isinstance(self._y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                self._temporal_encoding = self.temporal_encoding
            else:
                self._temporal_encoding = False
                warn(
                    "Temporal encoding has been disabled because the input data's "
                    "index is not a DatetimeIndex or PeriodIndex. Temporal encoding "
                    "only works with time-based indices. To disable this warning "
                    "set manually temporal_encoding=False when initializing the model."
                )

            from sktime.networks.ltsf.utils.timefeatures import get_mark_vocab_sizes

            self.mark_vocab_sizes = get_mark_vocab_sizes(
                temporal_encoding_type=self.temporal_encoding_type,
                freq=self.freq,
            )
        else:
            self._temporal_encoding = self.temporal_encoding
            self.mark_vocab_sizes = None

        return LTSFTransformerNetwork(
            seq_len=self.seq_len,
            context_len=self.context_len,
            pred_len=self._pred_len,
            output_attention=self.output_attention,
            mark_vocab_sizes=self.mark_vocab_sizes,
            position_encoding=self.position_encoding,
            temporal_encoding=self._temporal_encoding,
            temporal_encoding_type=self.temporal_encoding_type,
            enc_in=self.enc_in,
            dec_in=self.dec_in,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            factor=self.factor,
            dropout=self.dropout,
            activation=self.activation,
            c_out=self.c_out,
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
                "seq_len": 4,
                "context_len": 2,
                "pred_len": 2,
                "d_model": 16,
                "n_heads": 1,
                "d_ff": 32,
                "e_layers": 1,
                "d_layers": 1,
                "factor": 5,
                "dropout": 0.1,
                "activation": "relu",
                "freq": "h",
                "position_encoding": True,
                "temporal_encoding": True,
                "temporal_encoding_type": "linear",
                "num_epochs": 1,
                "batch_size": 1,
                "lr": 0.008,
            },
            {
                "seq_len": 4,
                "context_len": 2,
                "pred_len": 2,
                "d_model": 16,
                "n_heads": 1,
                "d_ff": 32,
                "e_layers": 1,
                "d_layers": 1,
                "factor": 5,
                "dropout": 0.1,
                "activation": "relu",
                "freq": "h",
                "position_encoding": False,
                "temporal_encoding": True,
                "temporal_encoding_type": "embed",
                "num_epochs": 1,
                "batch_size": 1,
                "lr": 0.008,
            },
        ]
        return params
