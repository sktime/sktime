"""Conditional Invertible Neural Network (cINN) for forecasting."""

__author__ = ["benHeid"]

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base.adapters._pytorch import (
    BaseDeepNetworkPyTorch,
    PyTorchTrainDataset,
)
from sktime.forecasting.trend import CurveFitForecaster
from sktime.networks.cinn import CINNNetwork
from sktime.transformations.merger import Merger
from sktime.transformations.series.fourier import FourierFeatures
from sktime.transformations.series.summarize import WindowSummarizer

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


def default_sine(x, amplitude, phase, offset, amplitude2, amplitude3, phase2):
    """Calculate a special sine for the cINN."""
    sbase = np.sin(x * (1 / 365 / 24 * np.pi * 4) + phase) * amplitude + offset
    s1 = (
        amplitude2
        * x
        * (np.sin(x * (1 / 365 / 24 * np.pi * 2) + (365 * 24 * np.pi) + phase2) + 1)
    )
    s2 = amplitude3 * x * (np.sin(x * (1 / 365 / 24 * np.pi * 2) + phase2) - 1)
    return sbase + s1 + s2


class CINNForecaster(BaseDeepNetworkPyTorch):
    """
    Conditional Invertible Neural Network (cINN) Forecaster.

    This forecaster uses a cINN to forecast the time series. The cINN learns a
    bijective mapping between the time series and a normal distributed latent
    space. The latent space is then sampled and transformed back to the time
    series space. The cINN is conditioned on statistical and fourier term based
    features of the time series and the provided exogenous features. This
    forecaster was applied in the BigDEAL challenge by the KIT-IAI team
    and is described in [1]_.

    Parameters
    ----------
    n_coupling_layers : int, optional (default=15)
        Number of coupling layers in the cINN.
    hidden_dim_size : int, optional (default=64)
        Number of hidden units in the subnet.
    sample_dim : int, optional (default=24)
        Dimension of the samples that the cINN is creating
    batch_size : int, optional (default=64)
        Batch size for the training.
    encoded_cond_size : int, optional (default=64)
        Dimension of the encoded condition.
    lr : float, optional (default=5e-4)
        Learning rate for the Adam optimizer.
    weight_decay : float, optional (default=1e-5)
        Weight decay for the Adam optimizer.
    sp_list : list of int, optional (default=[24])
        List of seasonal periods to use for the Fourier features.
    fourier_terms_list : list of int, optional (default=[1, 1])
        List of number of Fourier terms to use for the Fourier features.
    window_size : int, optional (default=24*30)
        Window size for calculating the rolling statistics using the
        WindowSummarizer.
    lag_feature: str, optional (default="mean")
        The rolling statistic that the WindowSummarizer should calculate.
    num_epochs : int, optional (default=50)
        Number of epochs to train the cINN.
    verbose : bool, optional (default=False)
        Whether to print the training progress.
    f_statistic : function, optional (default=default_sine)
        Function to use for forecasting the rolling statistic.
    init_param_f_statistic : list of float, optional (default=[1, 0, 0, 10, 1, 1])
        Initial parameters for the f_statistic function.
    deterministic : bool, optional (default=False)
        Whether to use a deterministic or stochastic cINN. Note, deterministic
        should only used for testing.
    patience : int, optional (default=5)
        Number of epochs to wait before stopping the training.
    delta : float, optional (default=0.0001)
        Minimum change in the validation loss to consider as an improvement.
    val_split : float, optional (default=0.2)
        Fraction of the data to use for validation.

    References
    ----------
    ..[1] Heidrich, B., Hertel, M., Neumann, O., Hagenmeyer, V., & Mikut, R.
          (2023). Using conditional Invertible Neural Networks to Perform Mid-
          Term Peak Load Forecasting. IET Smart Grid, Under Review

    Examples
    --------
    >>> from sktime.forecasting.conditional_invertible_neural_network import (
    ...     CINNForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> model = CINNForecaster(window_size=100) # doctest: +SKIP
    >>> model.fit(y) # doctest: +SKIP
    CINNForecaster(...)
    >>> y_pred = model.predict(fh=[1,2,3]) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["benheid"],
        "python_dependencies": ["FrEIA", "torch"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
    }

    def __init__(
        self,
        n_coupling_layers=10,
        hidden_dim_size=32,
        sample_dim=24,
        batch_size=64,
        encoded_cond_size=64,
        lr=5e-4,
        weight_decay=1e-5,
        sp_list=None,
        fourier_terms_list=None,
        window_size=24 * 30,
        num_epochs=50,
        verbose=False,
        f_statistic=None,
        init_param_f_statistic=None,
        deterministic=False,
        lag_feature="mean",
        patience=5,
        delta=0.0001,
        val_split=0.2,
    ):
        self.n_coupling_layers = n_coupling_layers
        self.hidden_dim_size = hidden_dim_size
        self.sample_dim = sample_dim
        self.sp_list = sp_list
        self._sp_list = sp_list if sp_list is not None else [24]
        self.verbose = verbose
        self.encoded_cond_size = encoded_cond_size
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.f_statistic = f_statistic
        self._f_statistic = f_statistic if f_statistic is not None else default_sine
        self.init_param_f_statistic = init_param_f_statistic
        self._init_param_f_statistic = (
            init_param_f_statistic
            if init_param_f_statistic is not None
            else [1, 0, 0, 10, 1, 1]
        )
        self.fourier_terms_list = fourier_terms_list
        self._fourier_terms_list = fourier_terms_list if fourier_terms_list else [1]
        self.deterministic = deterministic
        self.lag_feature = lag_feature
        self.patience = patience
        self.delta = delta
        self.val_split = val_split
        super().__init__(num_epochs, batch_size, lr=lr)

    def _fit(self, y, fh, X=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Raises
        ------
        ValueError
            If `self.window_size` is larger than the length of `y`.
        RuntimeError
            If curve fitting fails due to non-convergence.

        Returns
        -------
        self : reference to self
        """
        if self.window_size > len(y):
            raise ValueError(
                f"Invalid window_size: {self.window_size}. "
                "It must be less than or equal to the size of the training data "
                f"({len(y)})."
            )

        # Fit the rolling mean forecaster
        rolling_mean = WindowSummarizer(
            lag_feature={
                self.lag_feature: [[-self.window_size // 2, self.window_size // 2]]
            },
            truncate="fill",
        ).fit_transform(y)

        self.function = CurveFitForecaster(
            self._f_statistic,
            {"p0": self._init_param_f_statistic},
            normalise_index=True,
        )

        try:
            # Attempt to fit the function with rolling mean data.
            # This step can fail if the optimization process does not converge,
            # often leading to a "RuntimeError: Optimal parameters not found".
            self.function.fit(rolling_mean.dropna())
        except Exception as e:
            # Raise a detailed RuntimeError, preserving traceback.
            raise RuntimeError(
                "Curve fitting error. Please check the parameters and try again.\n"
                f"Window size: {self.window_size}\n"
                f"f_statistic: {self._f_statistic}\n"
                f"init_param_f_statistic: {self._init_param_f_statistic}"
            ) from e  # Preserve original traceback

        self.fourier_features = FourierFeatures(
            sp_list=self._sp_list, fourier_terms_list=self._fourier_terms_list
        )
        self.fourier_features.fit(y)

        split_index = int(len(y) * (1 - self.val_split))

        dataset = self._prepare_data(
            y[:split_index], X[:split_index] if X is not None else None
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_data_loader_nll = None
        if self.val_split > 0:
            val_dataset_nll = self._prepare_data(
                y[split_index:], X[split_index:] if X is not None else None
            )
            val_data_loader_nll = DataLoader(
                val_dataset_nll, shuffle=False, batch_size=len(val_dataset_nll)
            )

        self.network = self._build_network(None)

        self.optimizer = self._instantiate_optimizer()
        early_stopper = _EarlyStopper(patience=self.patience, min_delta=self.delta)

        # Fit the cINN
        for epoch in range(self.num_epochs):
            if not self._run_epoch(
                epoch,
                data_loader,
                val_data_loader_nll,
                early_stopper,
            ):
                break
        if val_data_loader_nll is not None:
            self.network = early_stopper._best_model
        dataset = self._prepare_data(y, X if X is not None else None)
        X, y = next(iter(DataLoader(dataset, shuffle=False, batch_size=len(dataset))))

        res = self.network(y, c=X.reshape((-1, self.sample_dim * self.n_cond_features)))
        self.z_ = res[0].detach().numpy()
        self.z_mean_ = self.z_.mean(axis=0)
        self.z_std_ = self.z_.std()

    def _build_network(self, fh):
        return CINNNetwork(
            horizon=self.sample_dim,
            cond_features=self.n_cond_features,
            encoded_cond_size=self.encoded_cond_size,
            num_coupling_layers=self.n_coupling_layers,
        ).build()

    def _run_epoch(self, epoch, data_loader, val_data_loader_nll, early_stopper):
        nll = None
        for i, _input in enumerate(data_loader):
            (c, x) = _input
            self.optimizer.zero_grad()

            z, log_j = self.network(x, c)
            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / self.sample_dim
            nll.backward()
            # TODO combine with forecast loss for backward?

            torch.nn.utils.clip_grad_norm_(self.network.trainable_parameters, 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 200 == 0:
                with torch.no_grad():
                    val_nll = -1
                    if val_data_loader_nll is not None:
                        c, x = next(iter(val_data_loader_nll))
                        z, log_j = self.network(x, c)
                        val_nll = (
                            torch.mean(z**2) / 2 - torch.mean(log_j) / self.sample_dim
                        )
                        if early_stopper.early_stop(
                            val_nll.detach().numpy(), self.network
                        ):
                            return False
                    if self.verbose:
                        print(epoch, i, nll.detach().numpy(), val_nll.detach().numpy())
        return True

    def _predict(self, X=None, fh=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        if fh is None:
            fh = self._fh
        if len(fh) < self.sample_dim:
            index = pd.Index(list(fh.to_absolute(self.cutoff))).union(self._y.index)
        else:
            index = list(fh.to_absolute(self.cutoff))
        if X is not None:
            X = X.combine_first(self._X).loc[index]
        if self.deterministic:
            np.random.seed(42)
        z = np.random.normal(self.z_mean_, self.z_std_, (len(index), self.sample_dim))
        z = pd.DataFrame(z, index=index)

        dataset = self._prepare_data(yz=z, X=X, z=z)
        X, z = next(iter(DataLoader(dataset, shuffle=False, batch_size=len(index))))

        res = self.network.reverse_sample(
            z, c=X.reshape((-1, self.sample_dim * self.n_cond_features))
        )

        result = Merger(stride=1).fit_transform(
            res.reshape((len(res), 1, self.sample_dim))
        )

        return pd.Series(result.values.reshape(-1), index=index, name=self._y.name).loc[
            list(fh.to_absolute(self.cutoff))
        ]

    def _prepare_data(self, yz, X, z=None):
        cal_features = self.fourier_features.transform(yz)
        statistics = self.function.predict(
            fh=ForecastingHorizon(yz.index, is_relative=False)
        )
        to_concatenate = (
            [X, cal_features, statistics.to_frame()]
            if X is not None
            else [
                cal_features,
                statistics.to_frame(),
            ]
        )
        X = pd.DataFrame(
            np.concatenate(to_concatenate, axis=-1),
            index=yz.index,
        )
        self.n_cond_features = len(X.columns)
        if z is None:
            dataset = PyTorchTrainDataset(yz, 0, fh=self.sample_dim, X=X)
        else:
            dataset = PyTorchCinnTestDataset(z, 0, fh=self.sample_dim, X=X)
        return dataset

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
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {
                "num_epochs": 1,
                "window_size": 2,
                "sample_dim": 5,
                "f_statistic": _test_function,
                "init_param_f_statistic": [1, 1],
                "deterministic": True,
                "val_split": 0.5,
            },
            {
                "f_statistic": _test_function,
                "window_size": 4,
                "sample_dim": 4,
                "hidden_dim_size": 512,
                "n_coupling_layers": 1,
                "init_param_f_statistic": [0, 0],
                "deterministic": True,
                "val_split": 0,
            },
        ]
        return params

    # Serialization methods is required since FrEIA is using non-picklable objects
    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:
        if `path` is None, returns an in-memory serialized self
        if `path` is a file, stores the zip with that name at the location.
        The contents of the zip file are:
        _metadata - contains class of self, i.e., type(self).
        _obj - serialized self. This class uses the default serialization (pickle).
        keras/ - model, optimizer and state stored inside this directory.
        history - serialized history object.


        Parameters
        ----------
        path : None or file location (str or Path)
            if None, self is saved to an in-memory object
            if file location, self is saved to that file location. For eg:
                path="estimator" then a zip file `estimator.zip` will be made at cwd.
                path="/home/stored/estimator" then a zip file `estimator.zip` will be
                stored in `/home/stored/`.

        serialization_format: str, default = "pickle"
            Module to use for serialization.
            The available options are present under
            `sktime.base._base.SERIALIZATION_FORMATS`. Note that non-default formats
            might require installation of other soft dependencies.

        Returns
        -------
        if `path` is None - in-memory serialized self
        if `path` is file location - ZipFile with reference to the file
        """
        tmp_network = None
        tmp_forecasters = None
        if hasattr(self, "network"):
            self._state_dict = self.network.state_dict()
            tmp_network = self.network
            del self.network
        if hasattr(self, "forecasters_"):
            self._stored_forecasters = []
            for forecaster in self.forecasters_.values[0, :]:
                self._stored_forecasters.append(
                    forecaster.save(serialization_format=serialization_format)
                )
            tmp_forecasters = self.forecasters_
            del self.forecasters_
        serial = super().save(path, serialization_format)
        if tmp_network is not None:
            self.network = tmp_network
        if tmp_forecasters is not None:
            self.forecasters_ = tmp_forecasters
        return serial

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial : 1st element of output of `cls.save(None)`

        Returns
        -------
        deserialized self resulting in output `serial`, of `cls.save(None)`
        """
        import pickle

        cinn_forecaster = pickle.loads(serial)
        if hasattr(cinn_forecaster, "_state_dict"):
            cinn_forecaster.network = CINNNetwork(
                horizon=cinn_forecaster.sample_dim,
                cond_features=cinn_forecaster.n_cond_features,
                encoded_cond_size=cinn_forecaster.encoded_cond_size,
                num_coupling_layers=cinn_forecaster.n_coupling_layers,
            ).build()
            cinn_forecaster.network.load_state_dict(cinn_forecaster._state_dict)
        if hasattr(cinn_forecaster, "_stored_forecasters"):
            forecasters = []
            for forecaster in cinn_forecaster._stored_forecasters:
                forecasters.append(forecaster[0].load_from_serial(forecaster[1]))
            cinn_forecaster.forecasters_ = pd.DataFrame([forecasters])
            cinn_forecaster.stored_forecasters = None
        cinn_forecaster._state_dict = None
        return cinn_forecaster

    @classmethod
    def load_from_path(cls, path):
        """Load object from file location.

        Parameters
        ----------
        serial : result of ZipFile(path).open("object)

        Returns
        -------
        deserialized self resulting in output at `path`, of `cls.save(path)`
        """
        from zipfile import ZipFile

        with ZipFile(path, "r") as file:
            cinn_forecaster = cls.load_from_serial(file.open("_obj").read())
        return cinn_forecaster


def _test_function(x, a, b):
    return a * x + b


class PyTorchCinnTestDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh=None, X=None):
        self.y = y.values
        self.X = X.values if X is not None else X
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return max(len(self.y) - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
        else:
            exog_data = tensor([])
        return (
            exog_data,
            from_numpy(self.y[i]).float(),
        )


class _EarlyStopper:
    """
    Early stopping for the cINN.

    Parameters
    ----------
    patience : int, optional (default=1)
        Number of epochs to wait before stopping the training.
    min_delta : float, optional (default=0)
        Minimum change in the validation loss to consider as an improvement.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = None

    def early_stop(self, validation_loss, model):
        """
        Check if the training should be stopped. And saves the current best model.

        Parameters
        ----------
        validation_loss : float
            Validation loss of the current epoch.
        model : torch.nn.Module
            Current model.
        """
        if (
            self.min_validation_loss is None
            or validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = validation_loss
            self.counter = 0
            self._best_model = deepcopy(model)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
