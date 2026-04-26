from __future__ import annotations

from numbers import Integral
from typing import Optional

import numpy as np

from tsbootstrap.base_bootstrap import (
    BaseDistributionBootstrap,
    BaseMarkovBootstrap,
    BaseResidualBootstrap,
    BaseSieveBootstrap,
    BaseStatisticPreservingBootstrap,
)
from tsbootstrap.markov_sampler import MarkovSampler
from tsbootstrap.time_series_simulator import TimeSeriesSimulator
from tsbootstrap.utils.odds_and_ends import generate_random_indices

# TODO: add a check if generated block is only one unit long
# TODO: ensure docstrings align with functionality
# TODO: test -- check len(returned_indices) == X.shape[0]
# TODO: ensure x is 2d only for var, otherwise 1d or 2d with 1 feature
# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap
# TODO: add test to fit_ar to ensure input lags, if list, are unique


# Fit, then resample residuals.
class WholeResidualBootstrap(BaseResidualBootstrap):
    """
    Whole Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to the entire time series,
    without any block structure. This is the most basic form of residual
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        model_type="ar",
        model_params: Optional[dict] = None,  # noqa: UP007
        order=None,
        save_models: bool = False,
    ):
        self._model_type = model_type

        super().__init__(
            n_bootstraps=n_bootstraps,
            rng=rng,
            model_type=model_type,
            model_params=model_params,
            order=order,
            save_models=save_models,
        )

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        self._fit_model(X=X, y=y)

        # Resample residuals
        resampled_indices = generate_random_indices(
            self.resids.shape[0], self.config.rng  # type: ignore
        )

        resampled_residuals = self.resids[resampled_indices]  # type: ignore
        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + resampled_residuals
        return [resampled_indices], [bootstrap_samples]


class BlockResidualBootstrap(BaseResidualBootstrap):
    """
    Block Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to blocks of the time series.
    The residuals are bootstrapped using the specified block structure and
    added to the fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BaseBlockBootstrap
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        block_bootstrap,
        n_bootstraps: Integral = 10,  # type: ignore
        model_type="ar",
        model_params=None,
        order=None,
        save_models: bool = False,
        rng=None,
    ) -> None:
        super().__init__(
            n_bootstraps=n_bootstraps,
            rng=rng,
            model_type=model_type,
            model_params=model_params,
            order=order,
            save_models=save_models,
        )
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and store residuals, fitted values, etc.
        BaseResidualBootstrap._fit_model(self, X=X, y=y)

        # Generate blocks of residuals
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids  # type: ignore
        )

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + np.concatenate(block_data, axis=0)
        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeMarkovBootstrap(BaseMarkovBootstrap):
    """
    Whole Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Markov
    bootstrapping. The residuals are fit to a Markov model, and then
    resampled using the Markov model. The resampled residuals are added to
    the fitted values to generate new samples.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    method : str, default="middle"
        The method to use for compressing the blocks.
        Must be one of "first", "middle", "last", "mean", "mode", "median",
        "kmeans", "kmedians", "kmedoids".
    apply_pca_flag : bool, default=False
        Whether to apply PCA to the residuals before fitting the HMM.
    pca : PCA, default=None
        The PCA object to use for applying PCA to the residuals.
    n_iter_hmm : Integral, default=10
        Number of iterations for fitting the HMM.
    n_fits_hmm : Integral, default=1
        Number of times to fit the HMM.
    blocks_as_hidden_states_flag : bool, default=False
        Whether to use blocks as hidden states.
    n_states : Integral, default=2
        Number of states for the HMM.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order
        for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA
        and (p, d, q, s) for SARIMAX. It is either a single Integral or a
        list of non-consecutive ints for AR, and an Integral for VAR and ARCH.
        If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag
        only chooses the best lag, not the best order, so for the tuple values,
        it only chooses the best p, not the best (p, o, q) or (p, d, q, s).
        The rest of the values are set to 0.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals and generate new samples by changing the random_seed.
    """

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and store residuals, fitted values, etc.
        self._fit_model(X=X, y=y)

        # Fit HMM to residuals, just once.
        random_seed = self.config.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.config.apply_pca_flag,
                pca=self.config.pca,
                n_iter_hmm=self.config.n_iter_hmm,
                n_fits_hmm=self.config.n_fits_hmm,
                method=self.config.method,  # type: ignore
                blocks_as_hidden_states_flag=self.config.blocks_as_hidden_states_flag,
                random_seed=random_seed,  # type: ignore
            )

            markov_sampler.fit(
                blocks=self.resids, n_states=self.config.n_states  # type: ignore
            )
            self.hmm_object = markov_sampler

        # Resample the fitted values using the HMM.
        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.config.rng.integers(0, 1000)  # type: ignore
        )[0]

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return [np.arange(X.shape[0])], [bootstrap_samples]


class BlockMarkovBootstrap(BaseMarkovBootstrap):
    """
    Block Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to blocks of the time series. The
    residuals are fit to a Markov model, then resampled using the specified
    block structure. The resampled residuals are added to the fitted values
    to generate new samples. This class is a combination of the
    `BlockResidualBootstrap` and `WholeMarkovBootstrap` classes.

    Parameters
    ----------
    block_bootstrap : BaseBlockBootstrap
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    method : str, default="middle"
        The method to use for compressing the blocks.
        Must be one of "first", "middle", "last", "mean", "mode", "median",
        "kmeans", "kmedians", "kmedoids".
    apply_pca_flag : bool, default=False
        Whether to apply PCA to the residuals before fitting the HMM.
    pca : PCA, default=None
        The PCA object to use for applying PCA to the residuals.
    n_iter_hmm : Integral, default=10
        Number of iterations for fitting the HMM.
    n_fits_hmm : Integral, default=1
        Number of times to fit the HMM.
    blocks_as_hidden_states_flag : bool, default=False
        Whether to use blocks as hidden states.
    n_states : Integral, default=2
        Number of states for the HMM.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order
        for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA
        and (p, d, q, s) for SARIMAX. It is either a single Integral or a
        list of non-consecutive ints for AR, and an Integral for VAR and ARCH.
        If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag
        only chooses the best lag, not the best order, so for the tuple values,
        it only chooses the best p, not the best (p, o, q) or (p, d, q, s).
        The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals, resample using blocks once, and generate new samples by changing the random_seed.
    """

    def __init__(
        self,
        block_bootstrap,
        n_bootstraps: Integral = 10,  # type: ignore
        method="middle",
        apply_pca_flag: bool = False,
        pca=None,
        n_iter_hmm: Integral = 10,  # type: ignore
        n_fits_hmm: Integral = 1,  # type: ignore
        blocks_as_hidden_states_flag: bool = False,
        n_states: Integral = 2,  # type: ignore
        model_type="ar",
        model_params=None,
        order=None,
        save_models: bool = False,
        rng=None,
    ) -> None:
        super().__init__(
            n_bootstraps=n_bootstraps,
            method=method,
            apply_pca_flag=apply_pca_flag,
            pca=pca,
            n_iter_hmm=n_iter_hmm,
            n_fits_hmm=n_fits_hmm,
            blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
            n_states=n_states,
            model_type=model_type,
            model_params=model_params,
            order=order,
            save_models=save_models,
            rng=rng,
        )
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and store residuals, fitted values, etc.
        super()._fit_model(X=X, y=y)

        # Generate blocks of residuals
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids  # type: ignore
        )

        random_seed = self.config.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.config.apply_pca_flag,
                pca=self.config.pca,
                n_iter_hmm=self.config.n_iter_hmm,
                n_fits_hmm=self.config.n_fits_hmm,
                method=self.config.method,  # type: ignore
                blocks_as_hidden_states_flag=self.config.blocks_as_hidden_states_flag,
                random_seed=random_seed,  # type: ignore
            )

            markov_sampler.fit(
                blocks=block_data, n_states=self.config.n_states
            )
            self.hmm_object = markov_sampler

        # Resample the fitted values using the HMM.
        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.config.rng.integers(0, 1000)  # type: ignore
        )[0]

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    """
    Whole Bias Corrected Bootstrap class for time series data.

    This class applies bias corrected bootstrapping to the entire time series,
    without any block structure. This is the most basic form of bias corrected
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        if self.statistic_X is None:
            self.statistic_X = self._calculate_statistic(X=X)

        # Resample residuals
        resampled_indices = generate_random_indices(
            X.shape[0], self.config.rng
        )
        bootstrapped_sample = X[resampled_indices]
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(bootstrapped_sample)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_sample_bias_corrected = bootstrapped_sample + bias
        return [resampled_indices], [bootstrap_sample_bias_corrected]


class BlockStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    """
    Block Bias Corrected Bootstrap class for time series data.

    This class applies bias corrected bootstrapping to blocks of the time series.
    The residuals are resampled using the specified block structure and added to
    the fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BaseBlockBootstrap
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    statistic : Callable, default=np.mean
        A callable function to compute the statistic that should be preserved.
    statistic_axis : Integral, default=0
        The axis along which the statistic should be computed.
    statistic_keepdims : bool, default=False
        Whether to keep the dimensions of the statistic or not.
    rng :  Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        block_bootstrap,
        n_bootstraps: Integral = 10,  # type: ignore
        statistic=None,
        statistic_axis: Integral = 0,  # type: ignore
        statistic_keepdims: bool = False,
        rng=None,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        statistic_config : BaseStatisticPreservingBootstrapConfig
            The configuration object for the bias corrected bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            statistic=statistic,
            statistic_axis=statistic_axis,
            statistic_keepdims=statistic_keepdims,
            rng=rng,
        )
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        if self.statistic_X is None:
            self.statistic_X = super()._calculate_statistic(X=X)
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(X=X)

        block_data_concat = np.concatenate(block_data, axis=0)
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(block_data_concat)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_samples = block_data_concat + bias
        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeDistributionBootstrap(BaseDistributionBootstrap):
    """
    Whole Distribution Bootstrap class for time series data.

    This class applies distribution bootstrapping to the entire time series,
    without any block structure. This is the most basic form of distribution
    bootstrapping. The residuals are fit to a distribution, and then
    resampled using the distribution. The resampled residuals are added to
    the fitted values to generate new samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    We either fit the distribution to the residuals once and generate new samples from the fitted distribution with a new random seed, or resample the residuals once and fit the distribution to the resampled residuals, then generate new samples from the fitted distribution with the same random seed n_bootstrap times.
    """

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and residuals
        self._fit_model(X=X, y=y)
        # Fit the specified distribution to the residuals
        if not self.config.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super()._fit_distribution(self.resids)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=X.shape[0],
                random_state=self.config.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, X.shape[0])], [bootstrap_samples]

        else:
            # Resample residuals
            resampled_indices = generate_random_indices(
                self.resids.shape[0], self.config.rng
            )
            resampled_residuals = self.resids[resampled_indices]
            resids_dist, resids_dist_params = super()._fit_distribution(
                resampled_residuals
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=X.shape[0],
                random_state=self.config.rng,
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + resampled_residuals
            return [resampled_indices], [bootstrap_samples]


class BlockDistributionBootstrap(BaseDistributionBootstrap):
    """
    Block Distribution Bootstrap class for time series data.

    This class applies distribution bootstrapping to blocks of the time series.
    The residuals are fit to a distribution, then resampled using the specified
    block structure. Then new residuals are generated from the fitted
    distribution and added to the fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BaseBlockBootstrap
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    distribution: str, default='normal'
        The distribution to use for generating the bootstrapped samples.
        Must be one of 'poisson', 'exponential', 'normal', 'gamma', 'beta',
        'lognormal', 'weibull', 'pareto', 'geometric', or 'uniform'.
    refit: bool, default=False
        Whether to refit the distribution to the resampled residuals for each
        bootstrap. If False, the distribution is fit once to the residuals and
        the same distribution is used for all bootstraps.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order
        for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA
        and (p, d, q, s) for SARIMAX. It is either a single Integral or a
        list of non-consecutive ints for AR, and an Integral for VAR and ARCH.
        If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag
        only chooses the best lag, not the best order, so for the tuple values,
        it only chooses the best p, not the best (p, o, q) or (p, d, q, s).
        The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    We either fit the distribution to the residuals once and generate new samples from the fitted distribution with a new random seed, or resample the residuals once and fit the distribution to the resampled residuals, then generate new samples from the fitted distribution with the same random seed n_bootstrap times.
    """

    def __init__(
        self,
        block_bootstrap,
        n_bootstraps: Integral = 10,  # type: ignore
        distribution: str = "normal",
        refit: bool = False,
        model_type="ar",
        model_params=None,
        order=None,
        save_models: bool = False,
        rng=None,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        distribution_config : BaseDistributionBootstrapConfig
            The configuration object for the distribution bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            distribution=distribution,
            refit=refit,
            save_models=save_models,
            order=order,
            model_type=model_type,
            model_params=model_params,
            rng=rng,
        )
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and residuals
        super()._fit_model(X=X, y=y)
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids
        )
        block_data_concat = np.concatenate(block_data, axis=0)
        # Fit the specified distribution to the residuals
        if not self.config.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super()._fit_distribution(block_data_concat)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.config.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, block_data_concat.shape[0])], [
                bootstrap_samples
            ]

        else:
            # Resample residuals
            resids_dist, resids_dist_params = super()._fit_distribution(
                block_data_concat
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.config.rng,
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Sieve
    bootstrapping. The residuals are fit to a second model, and then new
    samples are generated by adding the new residuals to the fitted values.

    Parameters
    ----------
    resids_model_type : str, default="ar"
        The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
    resids_order : Integral or list or tuple, default=None
        The order of the model to use for fitting the residuals. If None, the order is automatically determined.
    save_resids_models : bool, default=False
        Whether to save the fitted models for the residuals.
    kwargs_base_sieve : dict, default=None
        Keyword arguments to pass to the SieveBootstrap class.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        self._fit_model(X=X, y=y)
        self._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.config.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.config.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        return [np.arange(X.shape[0])], [simulated_samples]


class BlockSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to blocks of the time series.
    The residuals are fit to a second model, then resampled using the
    specified block structure. The new residuals are then added to the
    fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BaseBlockBootstrap
        The block bootstrap algorithm.
    resids_model_type : str, default="ar"
        The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
    resids_order : Integral or list or tuple, default=None
        The order of the model to use for fitting the residuals. If None, the order is automatically determined.
    save_resids_models : bool, default=False
        Whether to save the fitted models for the residuals.
    kwargs_base_sieve : dict, default=None
        Keyword arguments to pass to the SieveBootstrap class.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.

    Methods
    -------
    _init_ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def __init__(
        self,
        block_bootstrap,
        n_bootstraps: Integral = 10,  # type: ignore
        resids_model_type="ar",
        resids_order=None,
        save_resids_models: bool = False,
        kwargs_base_sieve=None,
        model_type="ar",
        model_params=None,
        order=None,
        save_models: bool = False,
        rng=None,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        sieve_config : BaseSieveBootstrapConfig
            The configuration object for the sieve bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            resids_model_type=resids_model_type,
            resids_order=resids_order,
            save_resids_models=save_resids_models,
            kwargs_base_sieve=kwargs_base_sieve,
            model_type=model_type,
            model_params=model_params,
            order=order,
            save_models=save_models,
            rng=rng,
        )
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        # Fit the model and residuals
        super()._fit_model(X=X, y=y)
        super()._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.config.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.config.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        resids_resids = self.X_fitted - simulated_samples
        (
            block_indices,
            resids_resids_resampled,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=resids_resids
        )
        resids_resids_resampled_concat = np.concatenate(
            resids_resids_resampled, axis=0
        )

        bootstrapped_samples = self.X_fitted + resids_resids_resampled_concat

        return block_indices, [bootstrapped_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}
