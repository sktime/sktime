"""Conditional Invertible Neural Network (cINN) for forecasting."""
__author__ = ["benHeid"]


import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.trend import CurveFitForecaster
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


class cINNNetwork(nn.Module):
    """
    Conditional Invertible Neural Network.

    Parameters
    ----------
    horizon : int
        Forecasting horizon.
    cond_features : int
        Number of features in the condition.
    encoded_cond_size : int
        Dimension of the encoded condition.
    num_coupling_layers : int
        Number of coupling layers in the cINN.
    hidden_dim_size : int
        Number of hidden units in the subnet.
    activation : torch.nn.Module
        Activation function to use in the subnet.
    """

    def __init__(
        self,
        horizon,
        cond_features,
        encoded_cond_size=64,
        num_coupling_layers=15,
        hidden_dim_size=64,
        activation=nn.ReLU,
    ) -> None:
        super(cINNNetwork, self).__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_features * horizon, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_cond_size),
        )
        self.hidden_dim_size = hidden_dim_size
        self.activation = activation
        self.cinn = self.build_inn(
            horizon, cond_features, encoded_cond_size, num_coupling_layers
        )
        self.trainable_parameters = [
            p for p in self.cinn.parameters() if p.requires_grad
        ]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
        if self.cond_net:
            self.trainable_parameters += list(self.cond_net.parameters())

    def build_inn(self, horizon, cond_features, encoded_cond_size, num_coupling_layers):
        """
        Build the cINN.

        Parameters
        ----------
        horizon : int
            Forecasting horizon.
        cond_features : int
            Number of features in the condition.
        encoded_cond_size : int
            Dimension of the encoded condition.
        num_coupling_layers : int
            Number of coupling layers in the cINN.
        """
        nodes = [Ff.InputNode(horizon)]

        cond = Ff.ConditionNode(encoded_cond_size)

        for k in range(num_coupling_layers):
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {
                        "subnet_constructor": self.create_subnet(
                            hidden_dim_size=self.hidden_dim_size,
                            activation=self.activation,
                        )
                    },
                    conditions=cond,
                )
            )
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
        return Ff.GraphINN(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def create_subnet(self, hidden_dim_size=32, activation=nn.ReLU):
        """Create a subnet for the cINN.

        Parameters
        ----------
        hidden_dim_size : int, optional (default=32)
            Number of hidden units in the subnet.
        activation : torch.nn.Module, optional (default=nn.ReLU)
            Activation function to use in the subnet.
        """

        def get_subnet(ch_in, ch_out):
            return nn.Sequential(
                nn.Linear(ch_in, hidden_dim_size),
                activation(),
                nn.Linear(hidden_dim_size, ch_out),
            )

        return get_subnet

    def forward(self, x, c, rev=False):
        """Forward pass through the cINN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, horizon).
        c : torch.Tensor
            Condition tensor of shape (batch_size, cond_features * horizon).
        rev : bool, optional (default=False)
            Whether to run the reverse pass.
        """
        if isinstance(x, np.ndarray):
            if isinstance(c, np.ndarray):
                c = self._calculate_condition(torch.from_numpy(c.astype("float32")))
            else:
                c = self._calculate_condition(c)
            z, jac = self.cinn(torch.from_numpy(x.astype("float32")), c=c, rev=rev)
        else:
            c = self._calculate_condition(c)
            z, jac = self.cinn(x.float(), c=c, rev=rev)
        return z, jac

    def _calculate_condition(self, c):
        if c is not None:
            c = self.cond_net(c.flatten(1))
        return c

    def reverse_sample(self, z, c):
        """
        Reverse sample from the cINN.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, horizon).
        c : torch.Tensor
            Condition tensor of shape (batch_size, cond_features * horizon).
        """
        c = self._calculate_condition(c)
        return self.cinn(z, c=c, rev=True)[0].detach().numpy()


class cINNForecaster(BaseForecaster):
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
        sp_list=[24],
        fourier_terms_list=[1, 1],
        window_size=28 * 24,
        epochs=50,
        verbose=False,
        f_statistic=default_sine,
        init_param_f_statistic=[1, 0, 0, 10, 1, 1],
    ):
        self.n_coupling_layers = n_coupling_layers
        self.hidden_dim_size = hidden_dim_size
        self.sample_dim = sample_dim
        self.sp_list = sp_list
        self.verbose = verbose
        self.epochs = epochs
        self.encoded_cond_size = encoded_cond_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.f_statistic = f_statistic
        self.init_param_f_statistic = init_param_f_statistic
        self.fourier_terms_list = fourier_terms_list
        # leave this as is
        super().__init__()

    def _fit(self, y, X, fh):
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
        # TODO future work might extend this features to use also min, max, etc.
        rolling_mean = WindowSummarizer(
            lag_feature={"mean": [[-self.window_size // 2, self.window_size // 2]]},
            truncate="fill",
        ).fit_transform(y)

        self.function = CurveFitForecaster(
            self.f_statistic, {"p0": self.init_param_f_statistic}
        )
        self.function.fit(rolling_mean.dropna())
        self.fourier_features = FourierFeatures(
            sp_list=self.sp_list, fourier_terms_list=self.fourier_terms_list
        )
        self.fourier_features.fit(y)

        y, X = self._prepare_data(y, X)
        dataset = TensorDataset(y, X)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.cinn = self.cINNNetwork(
            horizon=self.sample_dim,
            cond_features=self.n_cond_features,
            encoded_cond_size=self.encoded_cond_size,
            num_coupling_layers=self.n_coupling_layers,
        )
        self.optimizer = torch.optim.Adam(
            self.cinn.trainable_parameters, lr=self.lr, weight_decay=self.weight_decay
        )

        # Fit the cINN
        for epoch in range(self.epochs):
            self._run_epoch(epoch, data_loader)

        self.z_mean_ = 0
        self.z_std_ = 0.3

    def _run_epoch(self, epoch, data_loader):
        for i, _input in enumerate(data_loader):
            (x, c) = _input

            z, log_j = self.cinn(x, c)  # torch.cat([c, w], axis=-1))
            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / self.sample_dim
            nll.backward()

            torch.nn.utils.clip_grad_norm(self.cinn.trainable_parameters, 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if not i % 100 and self.verbose:
                print(epoch, i, nll.detach().numpy())
                pass
        return nll.detach().numpy()

    # todo: implement this, mandatory
    def _predict(self, fh, X):
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
        index = list(fh.to_absolute(self.cutoff))
        X = X.loc[index]
        z = np.random.normal(self.z_mean_, self.z_std_, (len(X)))
        z = pd.Series(z, index=index)

        z, X = self._prepare_data(z, X)

        res = self.cinn.reverse_sample(z, c=X).reshape((-1, self.sample_dim))

        result = Merger().fit_transform(res)

        return pd.DataFrame(result, index=index)

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

        Zt = []
        Xt = []

        for k in range(self.sample_dim):
            i = k
            j = len(X) - self.sample_dim + 1 + k
            Zt.append(yz[i:j])
            Xt.append(X[i:j])

        yz = np.stack(Zt).swapaxes(0, 1)
        X = np.stack(Xt).swapaxes(0, 1).swapaxes(1, 2)

        yz = torch.from_numpy(yz.astype("float32"))
        X = torch.from_numpy(X.astype("float32"))
        return yz, X

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # TODO Lukas Probably you should implement this method!

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
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

        # TODO
