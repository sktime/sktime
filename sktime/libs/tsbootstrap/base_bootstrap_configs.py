from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np
from scipy.stats import (
    beta,
    expon,
    gamma,
    geom,
    lognorm,
    norm,
    pareto,
    poisson,
    uniform,
    weibull_min,
)
from skbase.base import BaseObject
from sklearn.decomposition import PCA  # type: ignore

from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
    RngTypes,
)
from tsbootstrap.utils.validate import (
    validate_literal_type,
    validate_order,
    validate_rng,
    validate_single_integer,
)


class BaseTimeSeriesBootstrapConfig(BaseObject):
    """
    Base configuration class for time series bootstrapping.

    This class is a specialized configuration class that enables time series
    bootstrapping. It is not meant to be used directly, but rather to be
    inherited by other configuration classes. It contains the parameters
    that are common to all time series bootstrapping methods.
    """

    _tags = {"object_type": "config"}

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        """
        self.n_bootstraps = n_bootstraps
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        """Getter for rng."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """Setter for rng. Performs validation on assignment."""
        self._rng = validate_rng(value)

    @property
    def n_bootstraps(self) -> Integral:
        """Getter for n_bootstraps."""
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value) -> None:
        """Setter for n_bootstraps. Performs validation on assignment."""
        validate_single_integer(value, min_value=1)  # type: ignore
        self._n_bootstraps = value


class BaseResidualBootstrapConfig(BaseTimeSeriesBootstrapConfig):
    """
    Configuration class for BaseResidualBootstrap.

    This class is a specialized configuration class that enables residual
    time series bootstrapping.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        model_type: ModelTypesWithoutArch = "ar",
        order=None,
        save_models: bool = False,
        model_params=None,
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        model_type : str, default="ar"
            The model type to use. Must be one of "ar", "arima", "sarima",
            "var", or "arch".
        order : Integral or list or tuple, default=None
            The order of the model. If None, the best order is chosen via TSFitBestLag.
            If Integral, it is the lag order for AR, ARIMA, and SARIMA,
            and the lag order for ARCH. If list or tuple, the order is a
            tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
            It is either a single Integral or a list of non-consecutive ints for AR,
            and an Integral for VAR and ARCH. If None, the best order is chosen
            via TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
            not the best order, so for the tuple values, it only chooses the best p,
            not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
        save_models : bool, default=False
            Whether to save the fitted models.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        **kwargs
            Additional keyword arguments to pass to the TSFit model.

        Raises
        ------
        ValueError
            If model_type is not one of "ar", "arima", "sarima", "var", or "arch".

        Notes
        -----
        The model_type and order parameters are passed to TSFitBestLag, which
        chooses the best lag and order for the model. The best lag and order are
        then used to fit the model to the data. The residuals are then stored
        for use in the bootstrap.

        References
        ----------
        .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Residual_bootstrap
        """
        self._model_type = model_type
        self.model_type = model_type
        self.order = order
        self.save_models = save_models
        self.model_params = model_params

        super().__init__(n_bootstraps=n_bootstraps, rng=rng)

    @property
    def model_type(self) -> str:
        """Getter for model_type."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        """Setter for model_type. Performs validation on assignment."""
        value = value.lower()
        validate_literal_type(value, ModelTypesWithoutArch)  # type: ignore
        self._model_type = value

    @property
    def order(self) -> OrderTypes:
        """Getter for order."""
        return self._order

    @order.setter
    def order(self, value) -> None:
        """Setter for order. Performs validation on assignment."""
        validate_order(value)
        self._order = value

    @property
    def save_models(self) -> bool:
        """Getter for save_models."""
        return self._save_models

    @save_models.setter
    def save_models(self, value: bool) -> None:
        """Setter for save_models. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("save_models must be a boolean.")
        self._save_models = value


class BaseMarkovBootstrapConfig(BaseResidualBootstrapConfig):
    """
    Configuration class for BaseMarkovBootstrap.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca=None,
        n_iter_hmm: Integral = 10,  # type: ignore
        n_fits_hmm: Integral = 1,  # type: ignore
        blocks_as_hidden_states_flag: bool = False,
        n_states: Integral = 2,  # type: ignore
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
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
        **kwargs
            Additional keyword arguments to pass to the BaseResidualBootstrapConfig class,
            except for n_bootstraps and rng, which are passed directly to the parent BaseTimeSeriesBootstrapConfig class.
            See the documentation for BaseResidualBootstrapConfig for more information.
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng, **kwargs)
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.blocks_as_hidden_states_flag = blocks_as_hidden_states_flag
        self.n_states = n_states

    @property
    def method(self) -> str:
        """Getter for method."""
        return self._method

    @method.setter
    def method(self, value: BlockCompressorTypes) -> None:
        """Setter for method. Performs validation on assignment."""
        validate_literal_type(value, BlockCompressorTypes)  # type: ignore
        self._method = value.lower()

    @property
    def apply_pca_flag(self) -> bool:
        """Getter for apply_pca_flag."""
        return self._apply_pca_flag

    @apply_pca_flag.setter
    def apply_pca_flag(self, value: bool) -> None:
        """Setter for apply_pca_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("apply_pca_flag must be a boolean.")
        self._apply_pca_flag = value

    @property
    def pca(self):
        """Getter for pca."""
        return self._pca

    @pca.setter
    def pca(self, value) -> None:
        """Setter for pca. Performs validation on assignment."""
        if value is not None and not isinstance(value, PCA):
            raise TypeError("pca must be an instance of PCA.")
        self._pca = value

    @property
    def n_iter_hmm(self) -> Integral:
        """Getter for n_iter_hmm."""
        return self._n_iter_hmm

    @n_iter_hmm.setter
    def n_iter_hmm(self, value: Integral) -> None:
        """Setter for n_iter_hmm. Performs validation on assignment."""
        validate_single_integer(value, min_value=10)  # type: ignore
        self._n_iter_hmm = value

    @property
    def n_fits_hmm(self) -> Integral:
        """Getter for n_fits_hmm."""
        return self._n_fits_hmm

    @n_fits_hmm.setter
    def n_fits_hmm(self, value: Integral) -> None:
        """Setter for n_fits_hmm. Performs validation on assignment."""
        validate_single_integer(value, min_value=1)  # type: ignore
        self._n_fits_hmm = value

    @property
    def blocks_as_hidden_states_flag(self) -> bool:
        """Getter for blocks_as_hidden_states_flag."""
        return self._blocks_as_hidden_states_flag

    @blocks_as_hidden_states_flag.setter
    def blocks_as_hidden_states_flag(self, value: bool) -> None:
        """Setter for blocks_as_hidden_states_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("blocks_as_hidden_states_flag must be a boolean.")
        self._blocks_as_hidden_states_flag = value

    @property
    def n_states(self) -> Integral:
        """Getter for n_states."""
        return self._n_states

    @n_states.setter
    def n_states(self, value: Integral) -> None:
        """Setter for n_states. Performs validation on assignment."""
        validate_single_integer(value, min_value=2)  # type: ignore
        self._n_states = value


class BaseStatisticPreservingBootstrapConfig(BaseTimeSeriesBootstrapConfig):
    """
    Configuration class for BaseStatisticPreservingBootstrap.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        statistic: Callable = np.mean,
        statistic_axis: Integral = 0,  # type: ignore
        statistic_keepdims: bool = False,
    ):
        """
        Initialize self.

        Parameters
        ----------
        statistic : Callable, default=np.mean
            A callable function to compute the statistic that should be preserved.
        statistic_axis : Integral, default=0
            The axis along which the statistic should be computed.
        statistic_keepdims : bool, default=False
            Whether to keep the dimensions of the statistic or not.

        Raises
        ------
        ValueError
            If statistic is not a callable function.
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng)
        self.statistic = statistic
        self.statistic_axis = statistic_axis
        self.statistic_keepdims = statistic_keepdims

    @property
    def statistic(self) -> Callable:
        """Getter for statistic."""
        return self._statistic

    @statistic.setter
    def statistic(self, value: Callable) -> None:
        """Setter for statistic. Performs validation on assignment."""
        if not callable(value):
            raise TypeError("statistic must be a callable function.")
        self._statistic = value

    @property
    def statistic_axis(self) -> Integral:
        """Getter for statistic_axis."""
        return self._statistic_axis

    @statistic_axis.setter
    def statistic_axis(self, value: Integral) -> None:
        """Setter for statistic_axis. Performs validation on assignment."""
        validate_single_integer(value, min_value=0)  # type: ignore
        self._statistic_axis = value

    @property
    def statistic_keepdims(self) -> bool:
        """Getter for statistic_keepdims."""
        return self._statistic_keepdims

    @statistic_keepdims.setter
    def statistic_keepdims(self, value: bool) -> None:
        """Setter for statistic_keepdims. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("statistic_keepdims must be a boolean.")
        self._statistic_keepdims = value


class BaseDistributionBootstrapConfig(BaseResidualBootstrapConfig):
    """
    Configuration class for BaseDistributionBootstrap.
    """

    distribution_methods = {
        "poisson": poisson,
        "exponential": expon,
        "normal": norm,
        "gamma": gamma,
        "beta": beta,
        "lognormal": lognorm,
        "weibull": weibull_min,
        "pareto": pareto,
        "geometric": geom,
        "uniform": uniform,
    }

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        distribution: str = "normal",
        refit: bool = False,
        save_models=False,
        model_type: ModelTypesWithoutArch = "ar",
        **kwargs,
    ) -> None:
        """
        Initialize the BaseDistributionBootstrap class.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
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
        **kwargs
            Additional keyword arguments to pass to the BaseResidualBootstrapConfig class,
            except for n_bootstraps and rng, which are passed directly to the parent BaseTimeSeriesBootstrapConfig class.
            See the documentation for BaseResidualBootstrapConfig for more information.

        Notes
        -----
        The distribution is fit to the residuals using the `fit` method of the
        distribution object. The parameters of the distribution are then used to
        generate new residuals using the `rvs` method of the distribution object.
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            rng=rng,
            save_models=save_models,
            model_type=model_type,
            **kwargs,
        )

        if self.model_type == "var":
            raise ValueError(
                "model_type cannot be 'var' for distribution bootstrap, since we can only fit uni-variate distributions."
            )

        self.distribution = distribution
        self.refit = refit

    @property
    def distribution(self) -> str:
        """Getter for distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: str) -> None:
        """Setter for distribution. Performs validation on assignment."""
        validate_literal_type(value, self.distribution_methods)
        self._distribution = value.lower()

    @property
    def refit(self) -> bool:
        """Getter for refit."""
        return self._refit

    @refit.setter
    def refit(self, value: bool) -> None:
        """Setter for refit. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("refit must be a boolean.")
        self._refit = value


class BaseSieveBootstrapConfig(BaseResidualBootstrapConfig):
    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        resids_model_type: ModelTypes = "ar",
        resids_order=None,
        save_resids_models: bool = False,
        kwargs_base_sieve=None,
        model_type: ModelTypesWithoutArch = "ar",
        model_params=None,
        order: OrderTypes = None,
        **kwargs_base_residual,
    ) -> None:
        """
        Initialize the BaseSieveBootstrap class.

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
        **kwargs_base_residual
            Additional keyword arguments to pass to the BaseResidualBootstrapConfig class,
            except for n_bootstraps and rng, which are passed directly to the parent BaseTimeSeriesBootstrapConfig class.
            See the documentation for BaseResidualBootstrapConfig for more information.
        """
        self.resids_order = resids_order
        self.save_resids_models = save_resids_models
        self.kwargs_base_sieve = kwargs_base_sieve
        self.kwargs_base_residual = kwargs_base_residual

        kwargs_base_sieve = (
            {} if kwargs_base_sieve is None else kwargs_base_sieve
        )
        super().__init__(
            n_bootstraps=n_bootstraps,
            rng=rng,
            model_type=model_type,
            model_params=model_params,
            order=order,
            **kwargs_base_residual,
        )

        # this must happen after the super().__init__ call
        # because of strange property magic
        self.resids_model_type = resids_model_type

        if hasattr(self, "_model_type") and self.model_type == "var":
            self._resids_model_type = "var"
        else:
            self._resids_model_type = resids_model_type

        self.resids_order = resids_order
        self.save_resids_models = save_resids_models
        self.resids_model_params = kwargs_base_sieve

    @property
    def resids_model_type(self) -> str:
        return self._resids_model_type

    @resids_model_type.setter
    def resids_model_type(self, value: str) -> None:
        validate_literal_type(value, ModelTypes)
        value = value.lower()
        if value == "var" and self.model_type != "var":
            raise ValueError(
                "resids_model_type can be 'var' only if model_type is also 'var'."
            )
        self._resids_model_type = value

    @property
    def resids_order(self) -> OrderTypes:
        return self._resids_order

    @resids_order.setter
    def resids_order(self, value) -> None:
        """
        Set the order of residuals.

        Parameters
        ----------
        value : Integral or list or tuple
            The order value to be set. Must be a positive integer, or a list/tuple of positive integers.

        Raises
        ------
        TypeError
            If the value is not of the expected type (Integral, list, or tuple).
        ValueError
            If the value is an integral but is negative.
            If the value is a list/tuple and not all elements are positive integers.
        """
        validate_order(value)
        self._resids_order = value

    @property
    def save_resids_models(self) -> bool:
        return self._save_resids_models

    @save_resids_models.setter
    def save_resids_models(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("save_resids_models must be a boolean.")
        self._save_resids_models = value
