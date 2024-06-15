# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""
import functools
from typing import Any, Dict, List, Optional

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _PytorchForecastingAdapter,
)
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["XinyuWu"]


class PytorchForecastingTFT(_PytorchForecastingAdapter):
    """pytorch-forecasting Temporal Fusion Transformer model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting TFT model [1]_
        for example: {"lstm_layers": 3, "hidden_continuous_size": 10}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> l1 = data.index.get_level_values(1).map(lambda x: int(x[3:]))
    >>> X_train = data.loc[l1 < 190, ["c0", "c1"]]
    >>> y_train = data.loc[l1 < 190, "c2"].to_frame()
    >>> X_test = data.loc[l1 >= 180, ["c0", "c1"]]
    >>> y_test = data.loc[l1 >= 180, "c2"].to_frame()
    >>> len_levels = len(y_test.index.names)
    >>> y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    ...     lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    ... )
    >>> # define the model
    >>> model = PytorchForecastingTFT(
    ...     trainer_params={
    ...         "max_epochs": 5,  # for quick test
    ...         "limit_train_batches": 10,  # for quick test
    ...     },
    ... )
    >>> # fit and predict
    >>> model.fit(y=y_train, X=X_train, fh=fh)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    | Name                               | Type                            | Params
    ----------------------------------------------------------------------------------------
    0  | loss                               | QuantileLoss                    | 0
    1  | logging_metrics                    | ModuleList                      | 0
    2  | input_embeddings                   | MultiEmbedding                  | 0
    3  | prescalers                         | ModuleDict                      | 48
    4  | static_variable_selection          | VariableSelectionNetwork        | 0
    5  | encoder_variable_selection         | VariableSelectionNetwork        | 1.8 K
    6  | decoder_variable_selection         | VariableSelectionNetwork        | 1.2 K
    7  | static_context_variable_selection  | GatedResidualNetwork            | 1.1 K
    8  | static_context_initial_hidden_lstm | GatedResidualNetwork            | 1.1 K
    9  | static_context_initial_cell_lstm   | GatedResidualNetwork            | 1.1 K
    10 | static_context_enrichment          | GatedResidualNetwork            | 1.1 K
    11 | lstm_encoder                       | LSTM                            | 2.2 K
    12 | lstm_decoder                       | LSTM                            | 2.2 K
    13 | post_lstm_gate_encoder             | GatedLinearUnit                 | 544
    14 | post_lstm_add_norm_encoder         | AddNorm                         | 32
    15 | static_enrichment                  | GatedResidualNetwork            | 1.4 K
    16 | multihead_attn                     | InterpretableMultiHeadAttention | 676
    17 | post_attn_gate_norm                | GateAddNorm                     | 576
    18 | pos_wise_ff                        | GatedResidualNetwork            | 1.1 K
    19 | pre_output_gate_norm               | GateAddNorm                     | 576
    20 | output_layer                       | Linear                          | 119
    ----------------------------------------------------------------------------------------
    16.7 K    Trainable params
    0         Non-trainable params
    16.7 K    Total params
    0.067     Total estimated model params size (MB)
    Sanity Checking: |                                                                                       | 0/? [00:00<?, ?it/s]
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 11.19it/s, v_num=166, train_loss_step=0.754, val_loss=0.745, train_loss_epoch=0.737]`Trainer.fit` stopped: `max_epochs=5` reached.
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 10.83it/s, v_num=166, train_loss_step=0.754, val_loss=0.745, train_loss_epoch=0.737]
    PytorchForecastingTFT(trainer_params={'limit_train_batches': 10,
                                        'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    >>> print(y_test)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-01-01  5.261697
                2000-01-02  5.614349
                2000-01-03  6.619191
                2000-01-04  5.159320
                2000-01-05  7.590924
    ...                          ...
    h0_4 h1_199 2000-02-10  6.591850
                2000-02-11  5.619114
                2000-02-12  5.105312
                2000-02-13  5.185010
                2000-02-14  4.534434

    [4500 rows x 1 columns]
    >>> print(y_pred)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-02-15  5.310687
                2000-02-16  5.162195
                2000-02-17  5.157579
                2000-02-18  5.360476
                2000-02-19  5.308441
    ...                          ...
    h0_4 h1_199 2000-02-15  5.232810
                2000-02-16  5.282471
                2000-02-17  5.254453
                2000-02-18  5.181715
                2000-02-19  5.188011

    [500 rows x 1 columns]
    >>>

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingTFT",
        model_params: Optional[Dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingTFT"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import TemporalFusionTransformer

        return TemporalFusionTransformer

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingTFT") -> dict:
        """Get keyword parameters for the TFT class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return (
            {}
            if self.allowed_encoder_known_variable_names is None
            else {
                "allowed_encoder_known_variable_names\
                    ": self.allowed_encoder_known_variable_names,
            }
        )

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
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_val_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            # from pytorch_forecasting.metrics import QuantileLoss

            early_stop_callback = EarlyStopping(
                monitor="train_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        # "loss": QuantileLoss(),
                        # can not pass test_set_params and test_set_params_sklearn
                        # QuantileLoss() != QuantileLoss()
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_val_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params


class PytorchForecastingNBeats(_PytorchForecastingAdapter):
    """pytorch-forecasting NBeats model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"num_blocks": [5, 5], "widths": [128, 1024]}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split y data for train and test
    >>> l1 = data.index.get_level_values(1).map(lambda x: int(x[3:]))
    >>> y_train = data.loc[l1 < 190, "c2"].to_frame()
    >>> y_test = data.loc[l1 >= 180, "c2"].to_frame()
    >>> len_levels = len(y_test.index.names)
    >>> y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    ...     lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    ... )
    >>> # define the model
    >>> model = PytorchForecastingNBeats(
    ...     trainer_params={
    ...         "max_epochs": 5,  # for quick test
    ...         "limit_train_batches": 10,  # for quick test
    ...     },
    ... )
    >>> # fit and predict
    >>> model.fit(y=y_train, fh=fh)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    | Name            | Type       | Params
    -----------------------------------------------
    0 | loss            | MASE       | 0
    1 | logging_metrics | ModuleList | 0
    2 | net_blocks      | ModuleList | 1.6 M
    -----------------------------------------------
    1.6 M     Trainable params
    0         Non-trainable params
    1.6 M     Total params
    6.563     Total estimated model params size (MB)
    Sanity Checking: |                                                                                       | 0/? [00:00<?, ?it/s]
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.85it/s, v_num=164, train_loss_step=0.741, val_loss=0.747, train_loss_epoch=0.735]`Trainer.fit` stopped: `max_epochs=5` reached.
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.16it/s, v_num=164, train_loss_step=0.741, val_loss=0.747, train_loss_epoch=0.735]
    PytorchForecastingNBeats(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, y=y_test)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    >>> print(y_test)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-01-01  6.308914
                2000-01-02  3.471440
                2000-01-03  4.169305
                2000-01-04  5.990554
                2000-01-05  5.611347
    ...                          ...
    h0_4 h1_199 2000-02-10  6.448248
                2000-02-11  4.290731
                2000-02-12  5.494657
                2000-02-13  4.752948
                2000-02-14  5.243385

    [4500 rows x 1 columns]
    >>> print(y_pred)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-02-15  5.167375
                2000-02-16  5.178759
                2000-02-17  5.251082
                2000-02-18  5.331861
                2000-02-19  5.372994
    ...                          ...
    h0_4 h1_199 2000-02-15  5.005799
                2000-02-16  4.998720
                2000-02-17  5.031197
                2000-02-18  5.081184
                2000-02-19  5.113482

    [500 rows x 1 columns]
    >>>

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "ignores-exogeneous-X": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingNBeats",
        model_params: Optional[Dict[str, Any]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingNBeats"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import NBeats

        return NBeats

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingNBeats") -> dict:
        """Get keyword parameters for the NBeats class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return {}

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
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params


class PytorchForecastingDeepAR(_PytorchForecastingAdapter):
    """pytorch-forecasting DeepAR model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"cell_type": "GRU", "rnn_layers": 3}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> l1 = data.index.get_level_values(1).map(lambda x: int(x[3:]))
    >>> X_train = data.loc[l1 < 190, ["c0", "c1"]]
    >>> y_train = data.loc[l1 < 190, "c2"].to_frame()
    >>> X_test = data.loc[l1 >= 180, ["c0", "c1"]]
    >>> y_test = data.loc[l1 >= 180, "c2"].to_frame()
    >>> len_levels = len(y_test.index.names)
    >>> y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    ...     lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    ... )
    >>> # define the model
    >>> model = PytorchForecastingDeepAR(
    ...     trainer_params={
    ...         "max_epochs": 5,  # for quick test
    ...         "limit_train_batches": 10,  # for quick test
    ...     },
    ... )
    >>> # fit and predict
    >>> model.fit(y=y_train, X=X_train, fh=fh)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    | Name                   | Type                   | Params
    ------------------------------------------------------------------
    0 | loss                   | NormalDistributionLoss | 0
    1 | logging_metrics        | ModuleList             | 0
    2 | embeddings             | MultiEmbedding         | 0
    3 | rnn                    | LSTM                   | 1.5 K
    4 | distribution_projector | Linear                 | 22
    ------------------------------------------------------------------
    1.5 K     Trainable params
    0         Non-trainable params
    1.5 K     Total params
    0.006     Total estimated model params size (MB)
    Sanity Checking: |                                                                                       | 0/? [00:00<?, ?it/s]
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.85it/s, v_num=168, train_loss_step=1.630, val_loss=1.730, train_loss_epoch=1.690]`Trainer.fit` stopped: `max_epochs=5` reached.
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.74it/s, v_num=168, train_loss_step=1.630, val_loss=1.730, train_loss_epoch=1.690]
    PytorchForecastingDeepAR(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    >>> print(y_test)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-01-01  5.006716
                2000-01-02  5.197903
                2000-01-03  4.477552
                2000-01-04  4.751521
                2000-01-05  3.323994
    ...                          ...
    h0_4 h1_199 2000-02-10  5.590399
                2000-02-11  5.595445
                2000-02-12  4.915307
                2000-02-13  4.726925
                2000-02-14  5.482842

    [4500 rows x 1 columns]
    >>> print(y_pred)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-02-15  4.919366
                2000-02-16  4.862666
                2000-02-17  5.021425
                2000-02-18  4.934844
                2000-02-19  4.808967
    ...                          ...
    h0_4 h1_199 2000-02-15  5.150748
                2000-02-16  5.230827
                2000-02-17  5.123736
                2000-02-18  5.139505
                2000-02-19  5.121511

    [500 rows x 1 columns]
    >>>

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingDeepAR",
        model_params: Optional[Dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingDeepAR"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import DeepAR

        return DeepAR

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingDeepAR") -> dict:
        """Get keyword parameters for the DeepAR class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return (
            {}
            if self.allowed_encoder_known_variable_names is None
            else {
                "allowed_encoder_known_variable_names\
                    ": self.allowed_encoder_known_variable_names,
            }
        )

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
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "cell_type": "GRU",
                        "rnn_layers": 3,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "cell_type": "GRU",
                        "rnn_layers": 3,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params


class PytorchForecastingNHiTS(_PytorchForecastingAdapter):
    """pytorch-forecasting NHiTS model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"interpolation_mode": "nearest", "activation": "Tanh"}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingNHiTS
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> l1 = data.index.get_level_values(1).map(lambda x: int(x[3:]))
    >>> X_train = data.loc[l1 < 190, ["c0", "c1"]]
    >>> y_train = data.loc[l1 < 190, "c2"].to_frame()
    >>> X_test = data.loc[l1 >= 180, ["c0", "c1"]]
    >>> y_test = data.loc[l1 >= 180, "c2"].to_frame()
    >>> len_levels = len(y_test.index.names)
    >>> y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    ...     lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
    ... )
    >>> # define the model
    >>> model = PytorchForecastingNHiTS(
    ...     trainer_params={
    ...         "max_epochs": 5,  # for quick test
    ...         "limit_train_batches": 10,  # for quick test
    ...     },
    ... )
    >>> # fit and predict
    >>> model.fit(y=y_train, X=X_train, fh=fh)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    | Name            | Type           | Params
    ---------------------------------------------------
    0 | loss            | MASE           | 0
    1 | logging_metrics | ModuleList     | 0
    2 | embeddings      | MultiEmbedding | 0
    3 | model           | NHiTS          | 978 K
    ---------------------------------------------------
    978 K     Trainable params
    0         Non-trainable params
    978 K     Total params
    3.914     Total estimated model params size (MB)
    Sanity Checking: |                                                                                       | 0/? [00:00<?, ?it/s]
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.80it/s, v_num=170, train_loss_step=0.800, val_loss=0.840, train_loss_epoch=0.846]`Trainer.fit` stopped: `max_epochs=5` reached.
    Epoch 4: 100%|███████| 10/10 [00:00<00:00, 14.37it/s, v_num=170, train_loss_step=0.800, val_loss=0.840, train_loss_epoch=0.846]
    PytorchForecastingNHiTS(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    >>> print(y_test)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-01-01  8.184178
                2000-01-02  5.444128
                2000-01-03  5.992600
                2000-01-04  5.223143
                2000-01-05  6.191883
    ...                          ...
    h0_4 h1_199 2000-02-10  7.498591
                2000-02-11  5.910466
                2000-02-12  7.409602
                2000-02-13  4.670040
                2000-02-14  5.454403

    [4500 rows x 1 columns]
    >>> print(y_pred)
                                c2
    h0   h1     time
    h0_0 h1_180 2000-02-15  5.764410
                2000-02-16  5.826406
                2000-02-17  5.925301
                2000-02-18  5.792100
                2000-02-19  5.760923
    ...                          ...
    h0_4 h1_199 2000-02-15  5.376267
                2000-02-16  5.227071
                2000-02-17  5.070744
                2000-02-18  5.249713
                2000-02-19  5.047630

    [500 rows x 1 columns]
    >>>

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingNHiTS",
        model_params: Optional[Dict[str, Any]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingNHiTS"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import NHiTS

        return NHiTS

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingNHiTS") -> dict:
        """Get keyword parameters for the NHiTS class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        if "n_blocks" in self._model_params.keys():
            stacks = len(self._model_params["n_blocks"])
        else:
            stacks = 3  # default value in pytorch-forecasting
        if self._max_prediction_length == 1:
            # avoid the bug in https://github.com/jdb78/pytorch-forecasting/issues/1571
            return {
                "downsample_frequencies": [1] * stacks,
                "pooling_sizes": [1] * stacks,
            }
        return {}

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
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "interpolation_mode": "nearest",
                        "activation": "Tanh",
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "interpolation_mode": "nearest",
                        "activation": "Tanh",
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params
