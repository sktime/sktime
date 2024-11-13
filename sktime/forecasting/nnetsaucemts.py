# copyright: sktime developers, T. Moudiki, BSD-3-Clause License (see LICENSE file)
"""Interface to estimators from nnetsauce by Techtonique."""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sktime.forecasting.base import BaseForecaster


class NnetsauceMTS(BaseForecaster):
    """Forecasting with Quasi-Randomized networks (from nnetsauce).

    See https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas

    Parameters
    ----------
    regressor: object.
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict()). Default is sklearn's RidgeCV.

    n_hidden_features: int.
        number of nodes in the hidden layer.

    activation_name: str.
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'.

    a: float.
        hyperparameter for 'prelu' or 'elu' activation function.

    nodes_sim: str.
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'.

    bias: boolean.
        indicates if the hidden layer contains a bias term (True) or not
        (False).

    dropout: float.
        regularization parameter; (random) percentage of nodes dropped out
        of the training.

    direct_link: boolean.
        indicates if the original predictors are included (True) in model's
          fitting or not (False).

    n_clusters: int.
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
        no clustering).

    cluster_encode: bool.
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding.

    type_clust: str.
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm').

    type_scaling: a tuple of 3 strings.
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax').

    lags: int.
        number of lags used for each time series.

    type_pi: str.

        type of prediction interval; currently:

        - "gaussian": simple, fast, but: assumes stationarity of Gaussian in-sample
        residuals and independence in the multivariate case
        - "kde": based on Kernel Density Estimation of in-sample residuals
        - "bootstrap": based on independent bootstrap of in-sample residuals
        - "block-bootstrap": based on basic block bootstrap of in-sample residuals
        - "scp-kde": Sequential split conformal prediction with Kernel Density
        Estimation of calibrated residuals
        - "scp-bootstrap": Sequential split conformal prediction with independent
        bootstrap of calibrated residuals
        - "scp-block-bootstrap": Sequential split conformal prediction with basic
        block bootstrap of calibrated residuals
        - "scp2-kde": Sequential split conformal prediction with Kernel Density
        Estimation of standardized calibrated residuals
        - "scp2-bootstrap": Sequential split conformal prediction with independent
        bootstrap of standardized calibrated residuals
        - "scp2-block-bootstrap": Sequential split conformal prediction with basic
        block bootstrap of standardized calibrated residuals
        - based on copulas of in-sample residuals: 'vine-tll', 'vine-bb1',
        'vine-bb6', 'vine-bb7', 'vine-bb8', 'vine-clayton',
        'vine-frank', 'vine-gaussian', 'vine-gumbel', 'vine-indep', 'vine-joe',
        'vine-student'
        - 'scp-vine-tll', 'scp-vine-bb1', 'scp-vine-bb6', 'scp-vine-bb7',
        'scp-vine-bb8', 'scp-vine-clayton', 'scp-vine-frank', 'scp-vine-gaussian',
        'scp-vine-gumbel', 'scp-vine-indep', 'scp-vine-joe', 'scp-vine-student'
        - 'scp2-vine-tll', 'scp2-vine-bb1', 'scp2-vine-bb6', 'scp2-vine-bb7',
        'scp2-vine-bb8', 'scp2-vine-clayton', 'scp2-vine-frank', 'scp2-vine-gaussian',
          'scp2-vine-gumbel', 'scp2-vine-indep', 'scp2-vine-joe', 'scp2-vine-student'

    block_size: int.
        size of block for 'type_pi' in ("block-bootstrap", "scp-block-bootstrap",
        "scp2-block-bootstrap").
        Default is round(3.15*(n_residuals^1/3))

    replications: int.
        number of replications (if needed, for predictive simulation). Default
          is 'None'.

    kernel: str.
        the kernel to use for residuals density estimation (used for predictive
        simulation). Currently, either 'gaussian' or 'tophat'.

    agg: str.
        either "mean" or "median" for simulation of bootstrap aggregating

    seed: int.
        reproducibility seed for nodes_sim=='uniform' or predictive simulation.

    backend: str.
        "cpu" or "gpu" or "tpu".

    verbose: int.
        0: not printing; 1: printing

    show_progress: bool.
        True: progress bar when fitting each series; False: no progress bar when
        fitting each series

    """

    from nnetsauce import MTS as MTS0 # dangerous

    _tags = {
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "both",
        # fill in "univariate" or "both"
        #   "univariate": inner _fit, _predict, receives only single-column DataFrame
        #   "both": inner _predict gets pd.DataFrame series with any number of columns
        #
        # specify one or multiple authors and maintainers, only for sktime contribution
        "authors": ["thierrymoudiki"],  # authors, GitHub handles
        "maintainers": ["thierrymoudiki"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        #     if interfacing a 3rd party estimator, ensure to give credit to the
        #     authors of the interfaced estimator
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        #     for 3rd party interfaces, the scope is the sktime class only
        # remove maintainer tag if maintained by sktime core team
        #
        # do not change these:
        # (look at advanced templates if you think these should change)
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        regressor=RidgeCV(),
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        lags=1,
        type_pi="kde",
        block_size=None,
        replications=None,
        kernel="gaussian",
        agg="mean",
        seed=123,
        backend="cpu",
        verbose=0,
        show_progress=True,
    ):
        self.obj = self.regressor = regressor
        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.a = a
        self.nodes_sim = nodes_sim
        self.bias = bias
        self.dropout = dropout
        self.direct_link = direct_link
        self.n_clusters = n_clusters
        self.cluster_encode = cluster_encode
        self.type_clust = type_clust
        self.type_scaling = type_scaling
        self.lags = lags
        self.type_pi = type_pi
        self.block_size = block_size
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.seed = seed
        self.backend = backend
        self.verbose = verbose
        self.show_progress = show_progress

        from nnetsauce import MTS as MTS0

        self.fitter = MTS0(
            obj=self.obj,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            cluster_encode=self.cluster_encode,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            lags=self.lags,
            type_pi=self.type_pi,
            block_size=self.block_size,
            replications=self.replications,
            kernel=self.kernel,
            agg=self.agg,
            seed=self.seed,
            backend=self.backend,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self.fitter.fit(y)
        self.obj = self.fitter.obj
        return self

        # IMPORTANT: avoid side effects to y, X, fh
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
    def _predict(self, fh, X=None):
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
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        h = fh[-1]
        if self.replications is not None or self.type_pi == "gaussian":
            res = self.fitter.predict(h=h).mean
        else:
            res = self.fitter.predict(h=h)
        res.index = pd.to_datetime(res.index)
        res_array = res.to_numpy()  # Convert to NumPy array for slicing
        fh_indices = fh.to_numpy() if isinstance(fh, pd.Index) else np.asarray(fh)
        fh_indices -= 1
        filtered_res_array = res_array[fh_indices, :]
        filtered_res_df = pd.DataFrame(
            filtered_res_array,
            index=res.index[fh_indices],  # Corresponding indices based on `fh`
            columns=res.columns,  # Original columns
        )
        return filtered_res_df

        # IMPORTANT: avoid side effects to X, fh

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
        assert (
            self.replications is not None or self.type_pi == "gaussian"
        ), "must have self.replications is not None or self.type_pi == 'gaussian'"

        # prepare return data frame
        var_names = self.fitter.series_names
        index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        h = fh[-1]
        fh_indices = fh.to_numpy() if isinstance(fh, pd.Index) else np.asarray(fh)
        fh_indices -= 1

        for a in alpha:
            level = 100 * (1 - a)
            res = self.fitter.predict(h=h, level=level)
            res.lower.index = pd.to_datetime(res.lower.index)
            res.upper.index = pd.to_datetime(res.upper.index)
            res_lower_array = res.lower.to_numpy()
            res_upper_array = res.upper.to_numpy()
            res_lower_df = pd.DataFrame(
                res_lower_array[fh_indices, :],
                columns=res.mean.columns,
                index=res.lower.index[fh_indices],
            )
            res_upper_df = pd.DataFrame(
                res_upper_array[fh_indices, :],
                columns=res.mean.columns,
                index=res.lower.index[fh_indices],
            )
            for var_name in var_names:
                pred_quantiles[(var_name, a)] = res_lower_df.loc[:, var_name]
                pred_quantiles[(var_name, 1 - a)] = res_upper_df.loc[:, var_name]

        return pred_quantiles

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
        params = {"regressor": RidgeCV(),
        "n_hidden_features":5,
        "activation_name":"relu",
        "a":0.01,
        "nodes_sim":"sobol",
        "bias":True,
        "dropout":0,
        "direct_link":True,
        "n_clusters":2,
        "cluster_encode":True,
        "type_clust":"kmeans",
        "type_scaling":("std", "std", "std"),
        "lags":1,
        "type_pi":"kde",
        "block_size":None,
        "replications":None,
        "kernel":"gaussian",
        "agg":"mean",
        "seed":123,
        "backend":"cpu",
        "verbose":0,
        "show_progress":True}
        return params

class NnetsauceDeepMTS(NnetsauceMTS):
    """Forecasting with Deep Quasi-Randomized networks (from nnetsauce).

    See https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas

    Parameters
    ----------
    regressor: object.
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict()). Default is sklearn's RidgeCV. 

    n_layers: int.
        number of hidden layers.

    n_hidden_features: int.
        number of nodes in the hidden layer.

    activation_name: str.
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'.

    a: float.
        hyperparameter for 'prelu' or 'elu' activation function.

    nodes_sim: str.
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'.

    bias: boolean.
        indicates if the hidden layer contains a bias term (True) or not
        (False).

    dropout: float.
        regularization parameter; (random) percentage of nodes dropped out
        of the training.

    direct_link: boolean.
        indicates if the original predictors are included (True) in model's
          fitting or not (False).

    n_clusters: int.
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
        no clustering).

    cluster_encode: bool.
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding.

    type_clust: str.
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm').

    type_scaling: a tuple of 3 strings.
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax').

    lags: int.
        number of lags used for each time series.

    type_pi: str.

        type of prediction interval; currently:

        - "gaussian": simple, fast, but: assumes stationarity of Gaussian in-sample
        residuals and independence in the multivariate case
        - "kde": based on Kernel Density Estimation of in-sample residuals
        - "bootstrap": based on independent bootstrap of in-sample residuals
        - "block-bootstrap": based on basic block bootstrap of in-sample residuals
        - "scp-kde": Sequential split conformal prediction with Kernel Density
        Estimation of calibrated residuals
        - "scp-bootstrap": Sequential split conformal prediction with independent
        bootstrap of calibrated residuals
        - "scp-block-bootstrap": Sequential split conformal prediction with basic
        block bootstrap of calibrated residuals
        - "scp2-kde": Sequential split conformal prediction with Kernel Density
        Estimation of standardized calibrated residuals
        - "scp2-bootstrap": Sequential split conformal prediction with independent
        bootstrap of standardized calibrated residuals
        - "scp2-block-bootstrap": Sequential split conformal prediction with basic
        block bootstrap of standardized calibrated residuals
        - based on copulas of in-sample residuals: 'vine-tll', 'vine-bb1',
        'vine-bb6', 'vine-bb7', 'vine-bb8', 'vine-clayton',
        'vine-frank', 'vine-gaussian', 'vine-gumbel', 'vine-indep', 'vine-joe',
        'vine-student'
        - 'scp-vine-tll', 'scp-vine-bb1', 'scp-vine-bb6', 'scp-vine-bb7',
        'scp-vine-bb8', 'scp-vine-clayton', 'scp-vine-frank', 'scp-vine-gaussian',
        'scp-vine-gumbel', 'scp-vine-indep', 'scp-vine-joe', 'scp-vine-student'
        - 'scp2-vine-tll', 'scp2-vine-bb1', 'scp2-vine-bb6', 'scp2-vine-bb7',
        'scp2-vine-bb8', 'scp2-vine-clayton', 'scp2-vine-frank', 'scp2-vine-gaussian',
          'scp2-vine-gumbel', 'scp2-vine-indep', 'scp2-vine-joe', 'scp2-vine-student'

    block_size: int.
        size of block for 'type_pi' in ("block-bootstrap", "scp-block-bootstrap",
        "scp2-block-bootstrap").
        Default is round(3.15*(n_residuals^1/3))

    replications: int.
        number of replications (if needed, for predictive simulation). Default
          is 'None'.

    kernel: str.
        the kernel to use for residuals density estimation (used for predictive
        simulation). Currently, either 'gaussian' or 'tophat'.

    agg: str.
        either "mean" or "median" for simulation of bootstrap aggregating

    seed: int.
        reproducibility seed for nodes_sim=='uniform' or predictive simulation.

    backend: str.
        "cpu" or "gpu" or "tpu".

    verbose: int.
        0: not printing; 1: printing

    show_progress: bool.
        True: progress bar when fitting each series; False: no progress bar when
        fitting each series

    """
    
    from nnetsauce import DeepMTS as DeepMTS0 # dangerous

    _tags = {
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "both",
        # fill in "univariate" or "both"
        #   "univariate": inner _fit, _predict, receives only single-column DataFrame
        #   "both": inner _predict gets pd.DataFrame series with any number of columns
        #
        # specify one or multiple authors and maintainers, only for sktime contribution
        "authors": ["thierrymoudiki"],  # authors, GitHub handles
        "maintainers": ["thierrymoudiki"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        #     if interfacing a 3rd party estimator, ensure to give credit to the
        #     authors of the interfaced estimator
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        #     for 3rd party interfaces, the scope is the sktime class only
        # remove maintainer tag if maintained by sktime core team
        #
        # do not change these:
        # (look at advanced templates if you think these should change)
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        regressor=RidgeCV(),
        n_layers=1,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        lags=1,
        type_pi="kde",
        block_size=None,
        replications=None,
        kernel="gaussian",
        agg="mean",
        seed=123,
        backend="cpu",
        verbose=0,
        show_progress=True,
    ):
        self.obj = self.regressor = regressor
        self.n_layers = n_layers
        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.a = a
        self.nodes_sim = nodes_sim
        self.bias = bias
        self.dropout = dropout
        self.direct_link = direct_link
        self.n_clusters = n_clusters
        self.cluster_encode = cluster_encode
        self.type_clust = type_clust
        self.type_scaling = type_scaling
        self.lags = lags
        self.type_pi = type_pi
        self.block_size = block_size
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.seed = seed
        self.backend = backend
        self.verbose = verbose
        self.show_progress = show_progress

        self.fitter = DeepMTS0(
            obj=self.obj,
            n_layers=self.n_layers,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            cluster_encode=self.cluster_encode,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            lags=self.lags,
            type_pi=self.type_pi,
            block_size=self.block_size,
            replications=self.replications,
            kernel=self.kernel,
            agg=self.agg,
            seed=self.seed,
            backend=self.backend,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )
