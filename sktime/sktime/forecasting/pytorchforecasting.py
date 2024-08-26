# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""

import functools
from typing import Any, Optional

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _PytorchForecastingAdapter,
)
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["XinyuWu"]


class PytorchForecastingTFT(_PytorchForecastingAdapter):
    """pytorch-forecasting Temporal Fusion Transformer model.

    Parameters
    ----------
    model_params :  dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting TFT model [1]_
        for example: {"lstm_layers": 3, "hidden_continuous_size": 10}
    dataset_params : dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be inferred from data, so you do not have to pass them
    train_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}
    model_path: string (default=None)
        try to load a existing model without fitting. Calling the fit function is
        still needed, but no real fitting will be performed.
    random_log_path: bool (default=False)
        use random root directory for logging. This parameter is for CI test in
        Github action, not designed for end users.

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> x = data["c0", "c1"]
    >>> y = data["c2"].to_frame()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     x, y, test_size=0.2, train_size=0.8, shuffle=False
    ... )
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
    >>> model.fit(y=y_train, X=X_train, fh=fh) # doctest skip
    PytorchForecastingTFT(trainer_params={'limit_train_batches': 10,
                                        'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
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
        # "python_dependencies": "pytorch-forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch-forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingTFT",
        model_params: Optional[dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[list[str]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
        broadcasting: bool = False,
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
            broadcasting,
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
        if not _check_soft_dependencies("pytorch-forecasting", severity="none"):
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "model_params": {
                        "hidden_size": 8,
                        "lstm_layers": 1,
                        "log_interval": -1,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "hidden_size": 8,
                        "lstm_layers": 1,
                        "dropout": 0.1,
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_interval": -1,
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
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "log_interval": -1,
                        "hidden_size": 8,
                        "lstm_layers": 1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "hidden_size": 8,
                        "lstm_layers": 1,
                        "dropout": 0.1,
                        # "loss": QuantileLoss(),
                        # can not pass test_set_params and test_set_params_sklearn
                        # QuantileLoss() != QuantileLoss()
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting


class PytorchForecastingNBeats(_PytorchForecastingAdapter):
    """pytorch-forecasting NBeats model.

    Parameters
    ----------
    model_params :  dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"num_blocks": [5, 5], "widths": [128, 1024]}
    dataset_params : dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be inferred from data, so you do not have to pass them
    train_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}
    model_path: string (default=None)
        try to load a existing model without fitting. Calling the fit function is
        still needed, but no real fitting will be performed.
    random_log_path: bool (default=False)
        use random root directory for logging. This parameter is for CI test in
        Github action, not designed for end users.

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split y data for train and test
    >>> y_train, y_test = train_test_split(
    ...     data["c2"].to_frame(), test_size=0.2, train_size=0.8, shuffle=False
    ... )
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
    >>> model.fit(y=y_train, fh=fh) # doctest skip
    PytorchForecastingNBeats(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, y=y_test)
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
        # "python_dependencies": "pytorch-forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch-forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "ignores-exogeneous-X": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingNBeats",
        model_params: Optional[dict[str, Any]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
        broadcasting: bool = False,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
            broadcasting,
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
        if not _check_soft_dependencies("pytorch-forecasting", severity="none"):
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "model_params": {
                        "num_blocks": [1, 1],
                        "num_block_layers": [1, 1],
                        "widths": [8, 8],
                        "log_interval": -1,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "num_blocks": [1, 1],
                        "num_block_layers": [1, 1],
                        "widths": [8, 8],
                        "log_interval": -1,
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
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "num_blocks": [1, 1],
                        "num_block_layers": [1, 1],
                        "widths": [8, 8],
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "num_blocks": [1, 1],
                        "num_block_layers": [1, 1],
                        "widths": [8, 8],
                        "backcast_loss_ratio": 1.0,
                        "dropout": 0.2,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting


class PytorchForecastingDeepAR(_PytorchForecastingAdapter):
    """pytorch-forecasting DeepAR model.

    Parameters
    ----------
    model_params :  dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"cell_type": "GRU", "rnn_layers": 3}
    dataset_params : dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}
    model_path: string (default=None)
        try to load a existing model without fitting. Calling the fit function is
        still needed, but no real fitting will be performed.
    deterministic: bool (default=False)
        set seed before predict, so that it will give the same output for the same input
    random_log_path: bool (default=False)
        use random root directory for logging. This parameter is for CI test in
        Github action, not designed for end users.

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> x = data["c0", "c1"]
    >>> y = data["c2"].to_frame()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     x, y, test_size=0.2, train_size=0.8, shuffle=False
    ... )
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
    >>> model.fit(y=y_train, X=X_train, fh=fh) # doctest skip
    PytorchForecastingDeepAR(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
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
        # "python_dependencies": "pytorch-forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch-forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingDeepAR",
        model_params: Optional[dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[list[str]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        deterministic: bool = False,
        random_log_path: bool = False,
        broadcasting: bool = False,
    ) -> None:
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        self.deterministic = deterministic
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
            broadcasting,
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
        if not _check_soft_dependencies("pytorch-forecasting", severity="none"):
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "cell_type": "GRU",
                        "rnn_layers": 1,
                        "hidden_size": 3,
                        "enable_checkpointing": False,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                    "deterministic": True,  # to pass test_score
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "cell_type": "GRU",
                        "rnn_layers": 2,
                        "hidden_size": 3,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                    "deterministic": True,  # to pass test_score
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
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                    "deterministic": True,  # to pass test_score
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "cell_type": "GRU",
                        "rnn_layers": 3,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                    "deterministic": True,  # to pass test_score
                },
            ]

        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting


class PytorchForecastingNHiTS(_PytorchForecastingAdapter):
    """pytorch-forecasting NHiTS model.

    Parameters
    ----------
    model_params :  dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"interpolation_mode": "nearest", "activation": "Tanh"}
    dataset_params : dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}
    model_path: string (default=None)
        try to load a existing model without fitting. Calling the fit function is
        still needed, but no real fitting will be performed.
    random_log_path: bool (default=False)
        use random root directory for logging. This parameter is for CI test in
        Github action, not designed for end users.

    Examples
    --------
    >>> # import packages
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting import PytorchForecastingNHiTS
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import train_test_split
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
    ... )
    >>> # define forecast horizon
    >>> max_prediction_length = 5
    >>> fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
    >>> # split X, y data for train and test
    >>> x = data["c0", "c1"]
    >>> y = data["c2"].to_frame()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     x, y, test_size=0.2, train_size=0.8, shuffle=False
    ... )
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
    >>> model.fit(y=y_train, X=X_train, fh=fh) # doctest skip
    PytorchForecastingNHiTS(trainer_params={'limit_train_batches': 10,
                                            'max_epochs': 5})
    >>> y_pred = model.predict(fh, X=X_test, y=y_test)
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
        # "python_dependencies": "pytorch-forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch-forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingNHiTS",
        model_params: Optional[dict[str, Any]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
        broadcasting: bool = False,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
            broadcasting,
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
        if not _check_soft_dependencies("pytorch-forecasting", severity="none"):
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "model_params": {
                        "hidden_size": 8,
                        "n_blocks": [1, 1],
                        "n_layers": 1,
                        "log_interval": -1,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "interpolation_mode": "nearest",
                        "activation": "Tanh",
                        "hidden_size": 8,
                        "n_blocks": [1, 1],
                        "n_layers": 1,
                        "log_interval": -1,
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
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "hidden_size": 8,
                        "n_blocks": [1, 1],
                        "n_layers": 1,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                        "enable_checkpointing": False,
                        "logger": False,
                    },
                    "model_params": {
                        "interpolation_mode": "nearest",
                        "activation": "Tanh",
                        "hidden_size": 8,
                        "n_blocks": [1, 1],
                        "n_layers": 1,
                        "log_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting
