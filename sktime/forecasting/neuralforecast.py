# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from neuralforecast by Nixtla."""

import functools
from typing import Optional, Union

from sktime.forecasting.base.adapters._neuralforecast import (
    _SUPPORTED_LOCAL_SCALAR_TYPES,
    _NeuralForecastAdapter,
)
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["yarnabrina", "geetu040", "pranavvp16"]


class NeuralForecastRNN(_NeuralForecastAdapter):
    """NeuralForecast RNN model.

    Interface to ``neuralforecast.models.RNN`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.

    Multi Layer Elman RNN (RNN), with MLP decoder.
    The network has ``tanh`` or ``relu`` non-linearities, it is trained using
    ADAM stochastic gradient descent.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [4]_ from ``pandas``
        use int freq when using RangeIndex in ``y``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict
    input_size : int (default=-1)
        maximum sequence length for truncated train backpropagation

        default (-1) uses all history
    inference_input_size : int (default=-1)
        maximum sequence length for truncated inference

        default (-1) uses all history
    encoder_n_layers : int (default=2)
        number of layers for the RNN
    encoder_hidden_size : int (default=200)
        units for the RNN's hidden state size
    encoder_activation : str (default="tanh")
        type of RNN activation from ``tanh`` or ``relu``
    encoder_bias : bool (default=True)
        whether or not to use biases b_ih, b_hh within RNN units
    encoder_dropout : float (default=0.0)
        dropout regularization applied to RNN outputs
    context_size : int (default=10)
        size of context vector for each timestamp on the forecasting window
    decoder_hidden_size : int (default=200)
        size of hidden layer for the MLP decoder
    decoder_layers : int (default=2)
        number of layers for the MLP decoder
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [5]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [5]_
    max_steps : int (default=1000)
        maximum number of training steps
    learning_rate : float (default=1e-3)
        learning rate between (0, 1)
    num_lr_decays : int (default=-1)
        number of learning rate decays, evenly distributed across max_steps
    early_stop_patience_steps : int (default=-1)
        number of validation iterations before early stopping
    val_check_steps : int (default=100)
        number of training steps between every validation loss check
    batch_size : int (default=32)
        number of different series in each batch
    valid_batch_size : Optional[int] (default=None)
        number of different series in each validation and test batch
    scaler_type : str (default="robust")
        type of scaler for temporal inputs normalization
    random_seed : int (default=1)
        random_seed for pytorch initializer and numpy generators
    num_workers_loader : int (default=0)
        workers to be used by ``TimeSeriesDataLoader``
    drop_last_loader : bool (default=False)
        whether ``TimeSeriesDataLoader`` drops last non-full batch
    trainer_kwargs : dict (default=None)
        keyword trainer arguments inherited from PyTorch Lighning's trainer [6]_
    optimizer : pytorch optimizer (default=None) [7]_
        optimizer to use for training, if passed with None defaults to ``Adam``
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer
    broadcasting : bool (default=False)
        if True, a model will be fit per time series.
        Panels, e.g., multiindex data input, will be broadcasted to single series,
        and for each single series, one copy of this forecaster will be applied.
    lr_scheduler : pytorch learning rate scheduler (default=None) [9]_
        user specified lr_scheduler instead of the default choice ``StepLR`` [10]_
    lr_scheduler_kwargs : dict (default=None)
        list of parameters used by the user specified ``lr_scheduler``

    Notes
    -----
    * If ``loss`` is unspecified, MAE is used as the loss function for training.
    * Only ``futr_exog_list`` will be considered as exogenous variables.

    Examples
    --------
    >>>
    >>> # importing necessary libraries
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.neuralforecast import NeuralForecastRNN
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> # loading the Longley dataset and splitting it into train and test subsets
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # creating model instance configuring the hyperparameters
    >>> model = NeuralForecastRNN(  # doctest: +SKIP
    ...     "A-DEC", futr_exog_list=["ARMED", "POP"], max_steps=5
    ... )
    >>>
    >>> # fitting the model
    >>> model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])  # doctest: +SKIP
    Seed set to 1
    Epoch 4: 100%|█| 1/1 [00:00<00:00, 42.85it/s, v_num=870, train_loss_step=0.589,
    train_loss_epoc
    NeuralForecastRNN(freq='A-DEC', futr_exog_list=['ARMED', 'POP'], max_steps=5)
    >>>
    >>> # getting point predictions
    >>> model.predict(X=X_test)  # doctest: +SKIP
    Predicting DataLoader 0: 100%|██████████████████████████████████| 1/1 [00:00<00:00,
    198.64it/s]
    1959    66241.984375
    1960    66700.125000
    1961    66550.195312
    1962    67310.007812
    Freq: A-DEC, Name: TOTEMP, dtype: float64
    >>>

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.rnn.html#rnn
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4]
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [5] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [6]
    https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    .. [7] https://pytorch.org/docs/stable/optim.html
    .. [8] https://pytorch.org/docs/stable/optim.html#algorithms
    .. [9] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html
    .. [10] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["yarnabrina"],
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self: "NeuralForecastRNN",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[list[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_activation: str = "tanh",
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        trainer_kwargs: Optional[dict] = None,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        broadcasting: bool = False,
        lr_scheduler=None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.loss = loss
        self.valid_loss = valid_loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            broadcasting=broadcasting,
        )

        # initiate internal variables to avoid AttributeError in future
        self._trainer_kwargs = None
        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastRNN") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastRNN") -> str:
        """Set custom model name."""
        return "RNN"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastRNN"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.models import RNN

        return RNN

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastRNN") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._trainer_kwargs = (
            {} if self.trainer_kwargs is None else self.trainer_kwargs
        )

        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            "input_size": self.input_size,
            "inference_input_size": self.inference_input_size,
            "encoder_n_layers": self.encoder_n_layers,
            "encoder_hidden_size": self.encoder_hidden_size,
            "encoder_activation": self.encoder_activation,
            "encoder_bias": self.encoder_bias,
            "encoder_dropout": self.encoder_dropout,
            "context_size": self.context_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_lr_decays": self.num_lr_decays,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.random_seed,
            "num_workers_loader": self.num_workers_loader,
            "drop_last_loader": self.drop_last_loader,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "trainer_kwargs": self._trainer_kwargs,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("neuralforecast", severity="error")
            _check_soft_dependencies("torch", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                },
            ]
        else:
            from neuralforecast.losses.pytorch import SMAPE, QuantileLoss
            from torch.optim import Adam
            from torch.optim.lr_scheduler import ConstantLR

            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                    "lr_scheduler": ConstantLR,
                    "lr_scheduler_kwargs": {"factor": 0.5},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting


class NeuralForecastLSTM(_NeuralForecastAdapter):
    """NeuralForecast LSTM model.

    Interface to ``neuralforecast.models.LSTM`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.

    The Long Short-Term Memory Recurrent Neural Network (LSTM), uses a
    multilayer LSTM encoder and an MLP decoder.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [4]_ from ``pandas``
        use int freq when using RangeIndex in ``y``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict
    input_size : int (default=-1)
        maximum sequence length for truncated train backpropagation

        default (-1) uses all history
    inference_input_size : int (default=-1)
        maximum sequence length for truncated inference

        default (-1) uses all history
    encoder_n_layers : int (default=2)
        number of layers for the LSTM
    encoder_hidden_size : int (default=200)
        units for the LSTM hidden state size
    encoder_bias : bool (default=True)
        whether or not to use biases b_ih, b_hh within LSTM units
    encoder_dropout : float (default=0.0)
        dropout regularization applied to LSTM outputs
    context_size : int (default=10)
        size of context vector for each timestamp on the forecasting window
    decoder_hidden_size : int (default=200)
        size of hidden layer for the MLP decoder
    decoder_layers : int (default=2)
        number of layers for the MLP decoder
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [5]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [5]_
    max_steps : int (default=1000)
        maximum number of training steps
    learning_rate : float (default=1e-3)
        learning rate between (0, 1)
    num_lr_decays : int (default=-1)
        number of learning rate decays, evenly distributed across max_steps
    early_stop_patience_steps : int (default=-1)
        number of validation iterations before early stopping
    val_check_steps : int (default=100)
        number of training steps between every validation loss check
    batch_size : int (default=32)
        number of different series in each batch
    valid_batch_size : Optional[int] (default=None)
        number of different series in each validation and test batch
    scaler_type : str (default="robust")
        type of scaler for temporal inputs normalization
    random_seed : int (default=1)
        random_seed for pytorch initializer and numpy generators
    num_workers_loader : int (default=0)
        workers to be used by `TimeSeriesDataLoader`
    drop_last_loader : bool (default=False)
        whether `TimeSeriesDataLoader` drops last non-full batch
    trainer_kwargs : dict (default=None)
        keyword trainer arguments inherited from PyTorch Lighning's trainer [6]_
    optimizer : pytorch optimizer (default=None) [7]_
        optimizer to use for training, if passed with None defaults to ``Adam``
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer
    broadcasting : bool (default=False)
        if True, a model will be fit per time series.
        Panels, e.g., multiindex data input, will be broadcasted to single series,
        and for each single series, one copy of this forecaster will be applied.
    lr_scheduler : pytorch learning rate scheduler (default=None) [9]_
        user specified lr_scheduler instead of the default choice ``StepLR`` [10]_
    lr_scheduler_kwargs : dict (default=None)
        list of parameters used by the user specified ``lr_scheduler``

    Notes
    -----
    * If ``loss`` is unspecified, MAE is used as the loss function for training.
    * Only ``futr_exog_list`` will be considered as exogenous variables.

    Examples
    --------
    >>>
    >>> # importing necessary libraries
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.neuralforecast import NeuralForecastLSTM
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> # loading the Longley dataset and splitting it into train and test subsets
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # creating model instance configuring the hyperparameters
    >>> model = NeuralForecastLSTM(  # doctest: +SKIP
    ...     "A-DEC", futr_exog_list=["ARMED", "POP"], max_steps=5
    ... )
    >>>
    >>> # fitting the model
    >>> model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])  # doctest: +SKIP
    Seed set to 1
    Epoch 4: 100%|█| 1/1 [00:00<00:00, 42.85it/s, v_num=870, train_loss_step=0.589, train_loss_epoc
    NeuralForecastLSTM(freq='A-DEC', futr_exog_list=['ARMED', 'POP'], max_steps=5)
    >>>
    >>> # getting point predictions
    >>> model.predict(X=X_test)  # doctest: +SKIP
    Predicting DataLoader 0: 100%|██████████████████████████████████| 1/1 [00:00<00:00, 198.64it/s]
    1959    64083.226562
    1960    64426.304688
    1961    64754.886719
    1962    64889.496094
    Freq: A-DEC, Name: TOTEMP, dtype: float64
    >>>

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html#lstm
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [5] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [6] https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    .. [7] https://pytorch.org/docs/stable/optim.html
    .. [8] https://pytorch.org/docs/stable/optim.html#algorithms
    .. [9] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html
    .. [10] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["pranavvp16", "yarnabrina"],
        "maintainers": ["pranavvp16"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self: "NeuralForecastLSTM",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[list[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 0.001,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        trainer_kwargs: Optional[dict] = None,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        broadcasting: bool = False,
        lr_scheduler=None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.loss = loss
        self.valid_loss = valid_loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            broadcasting=broadcasting,
        )

        self._trainer_kwargs = None
        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastLSTM") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastLSTM") -> str:
        """Set custom model name."""
        return "LSTM"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastLSTM"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.models import LSTM

        return LSTM

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastLSTM") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._trainer_kwargs = (
            {} if self.trainer_kwargs is None else self.trainer_kwargs
        )

        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            "input_size": self.input_size,
            "inference_input_size": self.inference_input_size,
            "encoder_n_layers": self.encoder_n_layers,
            "encoder_hidden_size": self.encoder_hidden_size,
            "encoder_bias": self.encoder_bias,
            "encoder_dropout": self.encoder_dropout,
            "context_size": self.context_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_lr_decays": self.num_lr_decays,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.random_seed,
            "num_workers_loader": self.num_workers_loader,
            "drop_last_loader": self.drop_last_loader,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "trainer_kwargs": self._trainer_kwargs,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}

        """
        del parameter_set

        try:
            _check_soft_dependencies("neuralforecast", severity="error")
            _check_soft_dependencies("torch", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "encode_bias": False,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
            ]
        else:
            from neuralforecast.losses.pytorch import SMAPE, QuantileLoss
            from torch.optim import Adam
            from torch.optim.lr_scheduler import LinearLR

            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                    "lr_scheduler": LinearLR,
                    "lr_scheduler_kwargs": {"start_factor": 0.25, "end_factor": 0.75},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting


class NeuralForecastGRU(_NeuralForecastAdapter):
    """NeuralForecast GRU model.

    Interface to ``neuralforecast.models.GRU`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.

    Multi Layer Recurrent Network with Gated Units (GRU), and
    MLP decoder. The network has non-linear activation functions, it is trained
    using ADAM stochastic gradient descent. The network accepts static, historic
    and future exogenous data, flattens the inputs.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [4]_ from ``pandas``
        use int freq when using RangeIndex in ``y``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict
    input_size : int (default=-1)
        maximum sequence length for truncated train backpropagation

        default (-1) uses all history
    inference_input_size : int (default=-1)
        maximum sequence length for truncated inference

        default (-1) uses all history
    encoder_n_layers : int (default=2)
        number of layers for the GRU
    encoder_hidden_size : int (default=200)
        units for the GRU's hidden state size
    encoder_bias : bool (default=True)
        whether or not to use biases b_ih, b_hh within GRU units
    encoder_dropout : float (default=0.0)
        dropout regularization applied to GRU outputs
    context_size : int (default=10)
        size of context vector for each timestamp on the forecasting window
    decoder_hidden_size : int (default=200)
        size of hidden layer for the MLP decoder
    decoder_layers : int (default=2)
        number of layers for the MLP decoder
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [5]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [5]_
    max_steps : int (default=1000)
        maximum number of training steps
    learning_rate : float (default=1e-3)
        learning rate between (0, 1)
    num_lr_decays : int (default=-1)
        number of learning rate decays, evenly distributed across max_steps
    early_stop_patience_steps : int (default=-1)
        number of validation iterations before early stopping
    val_check_steps : int (default=100)
        number of training steps between every validation loss check
    batch_size : int (default=32)
        number of different series in each batch
    valid_batch_size : Optional[int] (default=None)
        number of different series in each validation and test batch
    scaler_type : str (default="robust")
        type of scaler for temporal inputs normalization
    random_seed : int (default=1)
        random_seed for pytorch initializer and numpy generators
    num_workers_loader : int (default=0)
        workers to be used by ``TimeSeriesDataLoader``
    drop_last_loader : bool (default=False)
        whether ``TimeSeriesDataLoader`` drops last non-full batch
    optimizer : pytorch optimizer (default=None) [7]_
        optimizer to use for training, if passed with None defaults to ``Adam``
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer
    lr_scheduler : pytorch learning rate scheduler (default=None) [9]_
        user specified lr_scheduler instead of the default choice ``StepLR`` [10]_
    lr_scheduler_kwargs : dict (default=None)
        list of parameters used by the user specified ``lr_scheduler``
    trainer_kwargs : dict (default=None)
        keyword trainer arguments inherited from PyTorch Lighning's trainer [6]_
    broadcasting : bool (default=False)
        if True, a model will be fit per time series.
        Panels, e.g., multiindex data input, will be broadcasted to single series,
        and for each single series, one copy of this forecaster will be applied.

    Notes
    -----
    * If ``loss`` is unspecified, MAE is used as the loss function for training.
    * Only ``futr_exog_list`` will be considered as exogenous variables.

    Examples
    --------
    >>>
    >>> # importing necessary libraries
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.neuralforecast import NeuralForecastGRU
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> # loading the Longley dataset and splitting it into train and test subsets
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # creating model instance configuring the hyperparameters
    >>> model = NeuralForecastGRU(  # doctest: +SKIP
    ...     "A-DEC", futr_exog_list=["ARMED", "POP"], max_steps=5
    ... )
    >>>
    >>> # fitting the model
    >>> model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])  # doctest: +SKIP
    Seed set to 1
    Epoch 4: 100%|████████████████████████████| 1/1 [00:00<00:00, 20.71it/s, v_num=0, train_loss_step=0.745, train_loss_epoch=0.745]
    NeuralForecastGRU(freq='A-DEC', futr_exog_list=['ARMED', 'POP'], max_steps=5)
    >>>
    >>> # getting point predictions
    >>> model.predict(X=X_test)  # doctest: +SKIP
    Predicting DataLoader 0: 100%|██████████████████████████████████| 1/1 [00:00<00:00,
    198.64it/s]
    1959    64516.003906
    1960    63966.054688
    1961    64432.593750
    1962    64210.609375
    Freq: A-DEC, Name: TOTEMP, dtype: float64
    >>>

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.gru.html#gru
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4]
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [5] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [6]
    https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    .. [7] https://pytorch.org/docs/stable/optim.html
    .. [8] https://pytorch.org/docs/stable/optim.html#algorithms
    .. [9] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html
    .. [10] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["yarnabrina"],
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self: "NeuralForecastGRU",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[list[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        broadcasting: bool = False,
    ):
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.loss = loss
        self.valid_loss = valid_loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            broadcasting=broadcasting,
        )

        # initiate internal variables to avoid AttributeError in future
        self._trainer_kwargs = None
        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastGRU") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastGRU") -> str:
        """Set custom model name."""
        return "GRU"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastGRU"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.models import GRU

        return GRU

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastGRU") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._trainer_kwargs = (
            {} if self.trainer_kwargs is None else self.trainer_kwargs
        )

        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            "input_size": self.input_size,
            "inference_input_size": self.inference_input_size,
            "encoder_n_layers": self.encoder_n_layers,
            "encoder_hidden_size": self.encoder_hidden_size,
            "encoder_bias": self.encoder_bias,
            "encoder_dropout": self.encoder_dropout,
            "context_size": self.context_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_lr_decays": self.num_lr_decays,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.random_seed,
            "num_workers_loader": self.num_workers_loader,
            "drop_last_loader": self.drop_last_loader,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "trainer_kwargs": self._trainer_kwargs,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("neuralforecast", severity="error")
            _check_soft_dependencies("torch", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                },
            ]
        else:
            from neuralforecast.losses.pytorch import SMAPE, QuantileLoss
            from torch.optim import Adam
            from torch.optim.lr_scheduler import ConstantLR

            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                    "lr_scheduler": ConstantLR,
                    "lr_scheduler_kwargs": {"factor": 0.5},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]

        params_broadcasting = [{**param, "broadcasting": True} for param in params]
        params_no_broadcasting = [{**param, "broadcasting": False} for param in params]

        return params_broadcasting + params_no_broadcasting


class NeuralForecastDilatedRNN(_NeuralForecastAdapter):
    """NeuralForecast DilatedRNN model.

    Interface to ``neuralforecast.models.DilatedRNN`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.

    Multi Layer Elman DilatedRNN (DilatedRNN), with MLP decoder.
    The network has ``tanh`` or ``relu`` non-linearities, it is trained using
    ADAM stochastic gradient descent.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [4]_ from ``pandas``
        use int freq when using RangeIndex in ``y``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict
    input_size : int (default=-1)
        maximum sequence length for truncated train backpropagation

        default (-1) uses all history
    inference_input_size : int (default=-1)
        maximum sequence length for truncated inference

        default (-1) uses all history
    cell_type : str (default="LSTM")
        type of RNN cell to use. can be one of the following:

        - 'GRU'
        - 'RNN'
        - 'LSTM'
        - 'ResLSTM'
        - 'AttentiveLSTM'
    dilations : list of int list (default=None)
        dilations between layers, by default set to ``[[1, 2], [4, 8]]``
    encoder_hidden_size : int (default=200)
        units for the DilatedRNN's hidden state size
    context_size : int (default=10)
        size of context vector for each timestamp on the forecasting window
    decoder_hidden_size : int (default=200)
        size of hidden layer for the MLP decoder
    decoder_layers : int (default=2)
        number of layers for the MLP decoder
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [5]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [5]_
    max_steps : int (default=1000)
        maximum number of training steps
    learning_rate : float (default=1e-3)
        learning rate between (0, 1)
    num_lr_decays : int (default=3)
        number of learning rate decays, evenly distributed across max_steps
    early_stop_patience_steps : int (default=-1)
        number of validation iterations before early stopping
    val_check_steps : int (default=100)
        number of training steps between every validation loss check
    batch_size : int (default=32)
        number of different series in each batch
    valid_batch_size : Optional[int] (default=None)
        number of different series in each validation and test batch
    step_size : int (default=1)
        step size between each window of temporal data
    scaler_type : str (default="robust")
        type of scaler for temporal inputs normalization
    random_seed : int (default=1)
        random_seed for pytorch initializer and numpy generators
    num_workers_loader : int (default=0)
        workers to be used by ``TimeSeriesDataLoader``
    drop_last_loader : bool (default=False)
        whether ``TimeSeriesDataLoader`` drops last non-full batch
    trainer_kwargs : dict (default=None)
        keyword trainer arguments inherited from PyTorch Lighning's trainer [6]_
    optimizer : pytorch optimizer (default=None) [7]_
        optimizer to use for training, if passed with None defaults to ``Adam``
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer
    broadcasting : bool (default=False)
        if True, a model will be fit per time series.
        Panels, e.g., multiindex data input, will be broadcasted to single series,
        and for each single series, one copy of this forecaster will be applied.
    lr_scheduler : pytorch learning rate scheduler (default=None) [9]_
        user specified lr_scheduler instead of the default choice ``StepLR`` [10]_
    lr_scheduler_kwargs : dict (default=None)
        list of parameters used by the user specified ``lr_scheduler``

    Notes
    -----
    * If ``loss`` is unspecified, MAE is used as the loss function for training.
    * Only ``futr_exog_list`` will be considered as exogenous variables.

    Examples
    --------
    >>>
    >>> # importing necessary libraries
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.neuralforecast import NeuralForecastDilatedRNN
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> # loading the Longley dataset and splitting it into train and test subsets
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # creating model instance configuring the hyperparameters
    >>> model = NeuralForecastDilatedRNN(  # doctest: +SKIP
    ...     "A-DEC", futr_exog_list=["ARMED", "POP"], max_steps=5
    ... )
    >>>
    >>> # fitting the model
    >>> model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])  # doctest: +SKIP
    Seed set to 1
    Epoch 4: 100%|████████████████████████████| 1/1 [00:00<00:00, 48.76it/s, v_num=2, train_loss_step=0.798, train_loss_epoch=0.798]
    NeuralForecastDilatedRNN(freq='A-DEC', futr_exog_list=['ARMED', 'POP'], max_steps=5)
    >>>
    >>> # getting point predictions
    >>> model.predict(X=X_test)  # doctest: +SKIP
    Predicting DataLoader 0: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 51.84it/s]
    1959    63867.414062
    1960    64041.445312
    1961    64116.046875
    1962    64220.585938
    Freq: A-DEC, Name: TOTEMP, dtype: float64
    >>>

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.dilated_rnn.html#dilatedrnn
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4]
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [5] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [6]
    https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    .. [7] https://pytorch.org/docs/stable/optim.html
    .. [8] https://pytorch.org/docs/stable/optim.html#algorithms
    .. [9] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html
    .. [10] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["yarnabrina"],
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self: "NeuralForecastDilatedRNN",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[list[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        cell_type: str = "LSTM",
        dilations: Optional[list[list[int]]] = None,
        encoder_hidden_size: int = 200,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: Optional[dict] = None,
        broadcasting: bool = False,
        trainer_kwargs: Optional[dict] = None,
    ):
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.cell_type = cell_type
        self.dilations = dilations
        self.encoder_hidden_size = encoder_hidden_size
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.loss = loss
        self.valid_loss = valid_loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.step_size = step_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            broadcasting=broadcasting,
        )

        # initiate internal variables to avoid AttributeError in future
        self._dilations = None
        self._trainer_kwargs = None
        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastDilatedRNN") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastDilatedRNN") -> str:
        """Set custom model name."""
        return "DilatedRNN"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastDilatedRNN"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.models import DilatedRNN

        return DilatedRNN

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastDilatedRNN") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._dilations = [[1, 2], [4, 8]] if self.dilations is None else self.dilations

        self._trainer_kwargs = (
            {} if self.trainer_kwargs is None else self.trainer_kwargs
        )

        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            "input_size": self.input_size,
            "inference_input_size": self.inference_input_size,
            "cell_type": self.cell_type,
            "dilations": self._dilations,
            "encoder_hidden_size": self.encoder_hidden_size,
            "context_size": self.context_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_lr_decays": self.num_lr_decays,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "step_size": self.step_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.random_seed,
            "num_workers_loader": self.num_workers_loader,
            "drop_last_loader": self.drop_last_loader,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "trainer_kwargs": self._trainer_kwargs,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("neuralforecast", severity="error")
            _check_soft_dependencies("torch", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                },
            ]
        else:
            from neuralforecast.losses.pytorch import SMAPE, QuantileLoss
            from torch.optim import Adam
            from torch.optim.lr_scheduler import ConstantLR

            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                    "lr_scheduler": ConstantLR,
                    "lr_scheduler_kwargs": {"factor": 0.5},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]
        params_broadcasting = [{**param, "broadcasting": True} for param in params]
        params_no_broadcasting = [{**param, "broadcasting": False} for param in params]

        return params_broadcasting + params_no_broadcasting


class NeuralForecastTCN(_NeuralForecastAdapter):
    """NeuralForecast TCN model.

    Interface to ``neuralforecast.models.TCN`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.

    Temporal Convolution Network (TCN), with MLP decoder.
    The historical encoder uses dilated skip connections to obtain efficient long memory,
    while the rest of the architecture allows for future exogenous alignment.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [4]_ from ``pandas``
        use int freq when using RangeIndex in ``y``

        default ("auto") interprets freq from ForecastingHorizon in ``fit``
    local_scaler_type : str (default=None)
        scaler to apply per-series to all features before fitting, which is inverted
        after predicting

        can be one of the following:

        - 'standard'
        - 'robust'
        - 'robust-iqr'
        - 'minmax'
        - 'boxcox'
    futr_exog_list : str list, (default=None)
        future exogenous variables
    verbose_fit : bool (default=False)
        print processing steps during fit
    verbose_predict : bool (default=False)
        print processing steps during predict
    input_size : int (default=-1)
        maximum sequence length for truncated train backpropagation

        default (-1) uses all history
    inference_input_size : int (default=-1)
        maximum sequence length for truncated inference

        default (-1) uses all history
    kernel_size : int (default=2)
        size of the convolving kernel
    dilations : int list (default=None)
        controls the temporal spacing between the kernel points
        also known as the à trous algorithm
        by default set to ``[1, 2, 4, 8, 16]``
    encoder_hidden_size : int (default=200)
        units for the TCN's hidden state size
    encoder_activation : str (default="ReLU")
        type of TCN activation from `tanh` or `relu`
    context_size : int (default=10)
        size of context vector for each timestamp on the forecasting window
    decoder_hidden_size : int (default=200)
        size of hidden layer for the MLP decoder
    decoder_layers : int (default=2)
        number of layers for the MLP decoder
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [5]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [5]_
    max_steps : int (default=1000)
        maximum number of training steps
    learning_rate : float (default=1e-3)
        learning rate between (0, 1)
    num_lr_decays : int (default=-1)
        number of learning rate decays, evenly distributed across max_steps
    early_stop_patience_steps : int (default=-1)
        number of validation iterations before early stopping
    val_check_steps : int (default=100)
        number of training steps between every validation loss check
    batch_size : int (default=32)
        number of different series in each batch
    valid_batch_size : Optional[int] (default=None)
        number of different series in each validation and test batch
    scaler_type : str (default="robust")
        type of scaler for temporal inputs normalization
    random_seed : int (default=1)
        random_seed for pytorch initializer and numpy generators
    num_workers_loader : int (default=0)
        workers to be used by ``TimeSeriesDataLoader``
    drop_last_loader : bool (default=False)
        whether ``TimeSeriesDataLoader`` drops last non-full batch
    optimizer : pytorch optimizer (default=None) [7]_
        optimizer to use for training, if passed with None defaults to ``Adam``
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer
    lr_scheduler : pytorch learning rate scheduler (default=None) [9]_
        user specified lr_scheduler instead of the default choice ``StepLR`` [10]_
    lr_scheduler_kwargs : dict (default=None)
        list of parameters used by the user specified ``lr_scheduler``
    trainer_kwargs : dict (default=None)
        keyword trainer arguments inherited from PyTorch Lighning's trainer [6]_
    broadcasting : bool (default=False)
        if True, a model will be fit per time series.
        Panels, e.g., multiindex data input, will be broadcasted to single series,
        and for each single series, one copy of this forecaster will be applied.

    Notes
    -----
    * If ``loss`` is unspecified, MAE is used as the loss function for training.
    * Only ``futr_exog_list`` will be considered as exogenous variables.

    Examples
    --------
    >>>
    >>> # importing necessary libraries
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.neuralforecast import NeuralForecastTCN
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> # loading the Longley dataset and splitting it into train and test subsets
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # creating model instance configuring the hyperparameters
    >>> model = NeuralForecastTCN(  # doctest: +SKIP
    ...     "A-DEC", futr_exog_list=["ARMED", "POP"], max_steps=5
    ... )
    >>>
    >>> # fitting the model
    >>> model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])  # doctest: +SKIP
    Seed set to 1
    Epoch 4: 100%|████████████████████████████| 1/1 [00:00<00:00, 48.19it/s, v_num=5, train_loss_step=0.833, train_loss_epoch=0.833]
    NeuralForecastTCN(freq='A-DEC', futr_exog_list=['ARMED', 'POP'], max_steps=5)
    >>>
    >>> # getting point predictions
    >>> model.predict(X=X_test)  # doctest: +SKIP
    Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 183.39it/s]
    1959    63611.593750
    1960    63528.542969
    1961    63529.167969
    1962    63718.667969
    Freq: A-DEC, Name: TOTEMP, dtype: float64
    >>>

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.tcn.html#tcn
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4]
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [5] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [6]
    https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    .. [7] https://pytorch.org/docs/stable/optim.html
    .. [8] https://pytorch.org/docs/stable/optim.html#algorithms
    .. [9] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html
    .. [10] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["yarnabrina"],
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self: "NeuralForecastTCN",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[list[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        kernel_size: int = 2,
        dilations: Optional[list[int]] = None,
        encoder_hidden_size: int = 200,
        encoder_activation: str = "ReLU",
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        broadcasting: bool = False,
    ):
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.loss = loss
        self.valid_loss = valid_loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            broadcasting=broadcasting,
        )

        # initiate internal variables to avoid AttributeError in future
        self._dilations = None
        self._trainer_kwargs = None
        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastTCN") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastTCN") -> str:
        """Set custom model name."""
        return "TCN"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastTCN"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.models import TCN

        return TCN

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastTCN") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._dilations = [1, 2, 4, 8, 16] if self.dilations is None else self.dilations

        self._trainer_kwargs = (
            {} if self.trainer_kwargs is None else self.trainer_kwargs
        )

        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            "input_size": self.input_size,
            "inference_input_size": self.inference_input_size,
            "kernel_size": self.kernel_size,
            "dilations": self._dilations,
            "encoder_hidden_size": self.encoder_hidden_size,
            "encoder_activation": self.encoder_activation,
            "context_size": self.context_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_lr_decays": self.num_lr_decays,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.random_seed,
            "num_workers_loader": self.num_workers_loader,
            "drop_last_loader": self.drop_last_loader,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "trainer_kwargs": self._trainer_kwargs,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("neuralforecast", severity="error")
            _check_soft_dependencies("torch", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                },
            ]
        else:
            from neuralforecast.losses.pytorch import SMAPE, QuantileLoss
            from torch.optim import Adam
            from torch.optim.lr_scheduler import ConstantLR

            params = [
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "max_steps": 4,
                    "trainer_kwargs": {"logger": False},
                    "lr_scheduler": ConstantLR,
                    "lr_scheduler_kwargs": {"factor": 0.5},
                },
                {
                    "freq": "auto",
                    "inference_input_size": 2,
                    "encoder_hidden_size": 2,
                    "decoder_hidden_size": 3,
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]

        params_broadcasting = [{**param, "broadcasting": True} for param in params]
        params_no_broadcasting = [{**param, "broadcasting": False} for param in params]

        return params_broadcasting + params_no_broadcasting


__all__ = [
    "NeuralForecastRNN",
    "NeuralForecastLSTM",
    "NeuralForecastGRU",
    "NeuralForecastDilatedRNN",
    "NeuralForecastTCN",
]
