# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from neuralforecast by Nixtla."""
import functools
from typing import Callable, List, Optional, Union

from sktime.forecasting.base.adapters._neuralforecast import (
    _SUPPORTED_HYPERPARAMETER_TUNING_BACKENDS,
    _SUPPORTED_LOCAL_SCALAR_TYPES,
    _NeuralForecastAdapter,
    _NeuralForecastAutoAdapter,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

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
        optimizer to use for training, if passed with None defaults to Adam
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer

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
    }

    def __init__(
        self: "NeuralForecastRNN",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[List[str]] = None,
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
        optimizer_kwargs: dict = None,
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
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
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
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]

        return params


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
        optimizer to use for training, if passed with None defaults to Adam
    optimizer_kwargs : dict (default=None) [8]_
        dict of parameters to pass to the user defined optimizer

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
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["pranavvp16"],
        "maintainers": ["pranavvp16"],
        # "python_dependencies": "neuralforecast"
        # inherited from _NeuralForecastAdapter
        # estimator type
        # --------------
        "python_dependencies": ["neuralforecast>=1.6.4"],
    }

    def __init__(
        self: "NeuralForecastLSTM",
        freq: Union[str, int] = "auto",
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[List[str]] = None,
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
        optimizer_kwargs: dict = None,
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
        self.trainer_kwargs = trainer_kwargs

        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
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
                    "loss": QuantileLoss(0.5),
                    "valid_loss": SMAPE(),
                    "max_steps": 4,
                    "val_check_steps": 2,
                    "trainer_kwargs": {"logger": False},
                    "optimizer": Adam,
                    "optimizer_kwargs": {"lr": 0.001},
                },
            ]

        return params


class NeuralForecastAutoTFT(_NeuralForecastAutoAdapter):
    """NeuralForecast AutoTFT model.

    Interface to ``neuralforecast.auto.AutoTFT`` [1]_
    through ``neuralforecast.NeuralForecast`` [2]_,
    from ``neuralforecast`` [3]_ by Nixtla.
    Implements automatic hyperparameter optimisation
    for ``neuralforecast.models.TFT`` [4]_ model.

    Parameters
    ----------
    freq : Union[str, int] (default="auto")
        frequency of the data, see available frequencies [1]_ from ``pandas``
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
    loss : pytorch module (default=None)
        instantiated train loss class from losses collection [2]_
    valid_loss : pytorch module (default=None)
        instantiated validation loss class from losses collection [2]_
    config : dict or callable (default=None)
        dictionary with ray.tune defined search space or function that takes an optuna
        trial and returns a configuration dict
    search_alg : ray.tune.search variant or optuna.sampler (default=None)
        for ray see tune search algorithms [3]_
        for optuna see parameter sampling strategies [4]_
    num_samples : int (default=10)
        number of hyperparameter optimization steps/samples
    cpus : int (default=None)
        number of cpus to use during optimization.
        only used with ray tune
    gpus : int (default=None)
        number of gpus to use during optimization, default all available.
        only used with ray tune.
    refit_with_val : bool
        refit of best model should preserve val_size
    verbose : bool
        track progress
    backend : str (default='ray')
        backend to use for searching the hyperparameter space

        can be one of the following:

        - 'ray'
        - 'optuna'
    callbacks : list of callables (default=None)
        list of functions to call during the optimization process
        see [5]_ for ray reference and [6]_ for optuna reference

    Notes
    -----
    * Only ``futr_exog_list`` will be considered as exogenous variables.
    * ``config`` overrides default hyperparameter space and only uses itself as the
        entire search space.
    * if ``search_alg`` is unspecified,
        ``ray.tune.search.basic_variant.BasicVariantGenerator(random_state=1)`` is used.
    * If ``cpus`` is unspecified, number of logical CPUs in the system is used. It is
        determined using ``os.cpu_count()``.
    * If ``gpus`` is unspecified, number of available GPUs is used. It is determined
        using ``torch.cuda.device_count()``.
    * If ``loss`` is unspecified, MAE is used as the loss function for training.

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/neuralforecast/models.html#autotft
    .. [2] https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast
    .. [3] https://github.com/Nixtla/neuralforecast/
    .. [4] https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html#tft
    .. [5] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    .. [6] https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html
    .. [5] https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html
    .. [6] https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
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
        "python_dependencies": ["neuralforecast>=1.7.0"],
    }

    def __init__(
        self: "NeuralForecastAutoTFT",
        freq: str,
        local_scaler_type: Optional[_SUPPORTED_LOCAL_SCALAR_TYPES] = None,
        futr_exog_list: Optional[List[str]] = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        loss=None,
        valid_loss=None,
        config: Optional[Union[Callable, dict]] = None,
        search_alg=None,
        num_samples: int = 10,
        cpus: Optional[int] = None,
        gpus: Optional[int] = None,
        refit_with_val: bool = False,
        verbose: bool = False,
        backend: _SUPPORTED_HYPERPARAMETER_TUNING_BACKENDS = "ray",
        callbacks=None,
    ) -> None:
        super().__init__(
            freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            cpus=cpus,
            gpus=gpus,
            refit_with_val=refit_with_val,
            verbose=verbose,
            backend=backend,
            callbacks=callbacks,
        )

        self._loss = None
        self._valid_loss = None

    @functools.cached_property
    def algorithm_exogenous_support(self: "NeuralForecastAutoTFT") -> bool:
        """Set support for exogenous features."""
        return True

    @functools.cached_property
    def algorithm_name(self: "NeuralForecastAutoTFT") -> str:
        """Set custom model name."""
        return "AutoTFT"

    @functools.cached_property
    def algorithm_class(self: "NeuralForecastAutoTFT"):
        """Import underlying NeuralForecast algorithm class."""
        from neuralforecast.auto import AutoTFT

        return AutoTFT

    @functools.cached_property
    def algorithm_parameters(self: "NeuralForecastAutoTFT") -> dict:
        """Get keyword parameters for the underlying NeuralForecast algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class

        Notes
        -----
        This method should not include following parameters:
        - future exogenous columns (``futr_exog_list``) - used from ``__init__``
        - historical exogenous columns (``hist_exog_list``) - not supported
        - statis exogenous columns (``stat_exog_list``) - not supported
        - custom model name (``alias``) - used from ``algorithm_name``
        """
        if self.loss:
            self._loss = self.loss
        else:
            from neuralforecast.losses.pytorch import MAE

            self._loss = MAE()

        if self.valid_loss:
            self._valid_loss = self.valid_loss

        return {
            **self.common_auto_algorithm_parameters,
            "loss": self._loss,
            "valid_loss": self._valid_loss,
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
        except ModuleNotFoundError:
            params = [
                {"freq": "D", "backend": "ray"},
                {"freq": "D", "backend": "optuna"},
            ]
        else:
            params = [
                {"freq": "D", "refit_with_val": True, "backend": "ray"},
                {"freq": "D", "refit_with_val": False, "backend": "optuna"},
            ]

        return params


__all__ = ["NeuralForecastAutoTFT", "NeuralForecastRNN", "NeuralForecastLSTM"]
