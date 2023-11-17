"""Conditional Invertible Neural Network (cINN) for forecasting."""
__author__ = ["benHeid"]

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base.adapters._pytorch import (
    BaseDeepNetworkPyTorch,
    PyTorchTrainDataset,
)
from sktime.forecasting.trend import CurveFitForecaster
from sktime.networks.cinn import cINNNetwork
from sktime.transformations.merger import Merger
from sktime.transformations.series.fourier import FourierFeatures
from sktime.transformations.series.summarize import WindowSummarizer


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


class cINNForecaster(BaseDeepNetworkPyTorch):
    """
    Conditional Invertible Neural Network Forecaster.

    Parameters
    ----------
    n_coupling_layers : int, optional (default=15)
        Number of coupling layers in the cINN.
    hidden_dim_size : int, optional (default=64)
        Number of hidden units in the subnet.
    sample_dim : int, optional (default=24)
        Dimension of the samples that the cINN is creating
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
    window_size : int, optional (default=28*24)
        Window size for the rolling mean transformer.
    epochs : int, optional (default=50)
        Number of epochs to train the cINN.
    verbose : bool, optional (default=False)
        Whether to print the training progress.
    f_statistic : function, optional (default=default_sine)
        Function to use for the rolling mean transformer.
    init_param_f_statistic : list of float, optional (default=[1, 0, 0, 10, 1, 1])
        Initial parameters for the f_statistic function.

    Examples
    --------
    >>> from sktime.forecasting.conditional_invertible_neural_network import (
    ...     cINNForecaster,
    ...)
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> model = cINNForecaster()
    >>> model.fit(y) # doctest: +SKIP
    cINNForecaster(...)
    >>> y_pred = model.predict(fh=[1,2,3]) # doctest: +SKIP
    >>> y_pred # doctest: +SKIP
    1961-01    515.456726
    1961-02    576.704712
    1961-03    559.859680
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_version": None,
        "python_dependencies": ["FrEIA", "torch"],
    }

    def __init__(
        self,
        n_coupling_layers,
        hidden_dim_size,
        sample_dim,
        encoded_cond_size=64,
        lr=5e-4,
        weight_decay=1e-5,
        sp_list=None,
        fourier_terms_list=None,
        window_size=28 * 24,
        epochs=50,
        verbose=False,
        f_statistic=default_sine,
        init_param_f_statistic=None,
    ):
        self.n_coupling_layers = n_coupling_layers
        self.hidden_dim_size = hidden_dim_size
        self.sample_dim = sample_dim
        self.sp_list = sp_list if sp_list is not None else [24]
        self.verbose = verbose
        self.epochs = epochs
        self.encoded_cond_size = encoded_cond_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.f_statistic = f_statistic
        self.init_param_f_statistic = (
            init_param_f_statistic
            if init_param_f_statistic is not None
            else [1, 0, 0, 10, 1, 1]
        )
        self.fourier_terms_list = fourier_terms_list if fourier_terms_list else [1, 1]
        super().__init__()

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

        Returns
        -------
        self : reference to self
        """
        # Fit the rolling mean forecaster
        rolling_mean = WindowSummarizer(
            lag_feature={"mean": [[-self.window_size // 2, self.window_size // 2]]},
            truncate="fill",
        ).fit_transform(y)

        self.function = CurveFitForecaster(
            self.f_statistic, {"p0": self.init_param_f_statistic}, normalise_index=True
        )
        self.function.fit(rolling_mean.dropna())
        self.fourier_features = FourierFeatures(
            sp_list=self.sp_list, fourier_terms_list=self.fourier_terms_list
        )
        self.fourier_features.fit(y)

        dataset = self._prepare_data(y, X)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.network = self._build_network(None)

        self.optimizer = self._instatiate_optimizer()

        # Fit the cINN
        for epoch in range(self.epochs):
            self._run_epoch(epoch, data_loader)

        self.z_mean_ = 0
        self.z_std_ = 0.3

    def _build_network(self, fh):
        return cINNNetwork(
            horizon=self.sample_dim,
            cond_features=self.n_cond_features,
            encoded_cond_size=self.encoded_cond_size,
            num_coupling_layers=self.n_coupling_layers,
        ).build()

    def _run_epoch(self, epoch, data_loader):
        nll = None
        for _input in data_loader:
            (c, x) = _input

            z, log_j = self.network(x, c)  # torch.cat([c, w], axis=-1))
            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / self.sample_dim
            nll.backward()

            torch.nn.utils.clip_grad_norm(self.network.trainable_parameters, 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        #            if not i % 100 and self.verbose:
        #                print(epoch, i, nll.detach().numpy())
        #                pass
        return nll.detach().numpy()

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

        index = list(fh.to_absolute(self.cutoff))
        X = X.loc[index]
        z = np.random.normal(self.z_mean_, self.z_std_, (len(X)))
        z = pd.Series(z, index=index)

        dataset = self._prepare_data(z, X)
        X, z = next(iter(DataLoader(dataset, shuffle=False, batch_size=len(X))))

        res = self.network.reverse_sample(
            z, c=X.reshape((-1, self.sample_dim * self.n_cond_features))
        )

        result = Merger(stride=1).fit_transform(res.reshape((len(res), 1, 24)))

        return pd.DataFrame(result.values, index=index)

    def _prepare_data(self, yz, X):
        cal_features = self.fourier_features.transform(yz)
        statistics = self.function.predict(
            fh=ForecastingHorizon(yz.index, is_relative=False)
        )
        X = pd.DataFrame(
            np.concatenate([X, cal_features, statistics.to_frame()], axis=-1),
            index=yz.index,
        )
        self.n_cond_features = len(X.columns)

        dataset = PyTorchTrainDataset(yz, 0, fh=self.sample_dim, X=X)
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
            {},
        ]
        return params
