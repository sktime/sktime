# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from prophetverse."""

__author__ = ["felipeangelimvieira"]  # fkiraly for adapter

import pandas as pd

from sktime.forecasting.base._delegate import _DelegatedForecaster


def placeholder(cls):
    """Delegate to prophetverse if installed, otherwise use placeholder.

    If prophetverse 0.3 or higher is installed, this will directly
    return the forecaster imported from prophetverse.
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    try:
        if _check_soft_dependencies("prophetverse>=0.3.0", severity="none"):
            from prophetverse.sktime import Prophetverse

            return Prophetverse
    except Exception:  # noqa: S110
        pass

    # else we return the placeholder, which is a delegator
    return cls


@placeholder
class Prophetverse(_DelegatedForecaster):
    """Univariate prophetverse forecaster - prophet model implemented in numpyro.

    Estimator from the ``prophetverse`` package by ``felipeangelimvieira``.

    Differences to facebook's prophet:

    * logistic trend. Here, another parametrization is considered,
      and the capacity is not passed as input, but inferred from the data.

    * the users can pass arbitrary ``sktime`` transformers as ``feature_transformer``,
      for instance ``FourierFeatures`` or ``HolidayFeatures``.

    * no default weekly_seasonality/yearly_seasonality, this is left to the user
      via the ``feature_transformer`` parameter

    * Uses ``changepoint_interval`` instead of ``n_changepoints`` to set changepoints.

    * accepts configurations where each exogenous variable has a different function
      relating it to its additive effect on the time series.
      One can, for example, set different priors for a group of feature,
      or use a Hill function to model the effect of a feature.

    Parameters
    ----------
    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, ca nbe positive or negative.
          Absolute value must be less than number of training points.
          The range will be that number of points.
          A negative int indicates number of points
          counting from the end of the history, a positive int from the beginning.

    changepoint_prior_scale : float, optional, default=0.001
        Regularization parameter controlling the flexibility
        of the automatic changepoint selection.

    offset_prior_scale : float, optional, default=0.1
        Scale parameter for the prior distribution of the offset.
        The offset is the constant term in the piecewise trend equation.

    feature_transformer : sktime transformer, BaseTransformer, optional, default=None
        Transformer object to generate Fourier terms, holiday or other features.
        If None, no additional features are used.
        For multiple features, pass a ``FeatureUnion`` object with the transformers.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    noise_scale : float, optional, default=0.05
        Scale parameter for the observation noise.
    trend : str, optional, one of "linear" (default) or "logistic"
        Type of trend to use. Can be "linear" or "logistic".

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of MCMC warmup steps. Also known as burn-in.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains to run in parallel.

    inference_method : str, optional, one of "mcmc" or "map", default="map"
        Inference method to use. Can be "mcmc" or "map".

    optimizer_name : str, optional, default="Adam"
        Name of the numpyro optimizer to use for variational inference.

    optimizer_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the numpyro optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps to perform for variational inference.

    exogenous_effects : List[AbstractEffect], optional, default=None
        A list of ``prophetverse`` ``AbstractEffect`` objects
        defining the exogenous effects to be used in the model.

    default_effect : AbstractEffectm optional, default=None
        The default effect to be used when no effect is specified for a variable.

    default_exogenous_prior : tuple, default=None
        Default prior distribution for exogenous effects.

    rng_key : jax.random.PRNGKey or None (default
        Random number generator key.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse",
        # estimator type
        # --------------
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "enforce_index_type": [pd.Period, pd.DatetimeIndex],
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "_delegate"

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="linear",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.mcmc_chains = mcmc_chains
        self.inference_method = inference_method
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_steps = optimizer_steps
        self.exogenous_effects = exogenous_effects
        self.default_effect = default_effect
        self.rng_key = rng_key
        self.scale = scale

        super().__init__()

        # delegation, only for prophetverse 0.2.X
        from prophetverse.sktime import Prophet

        self._delegate = Prophet(**self.get_params())
