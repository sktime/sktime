# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from prophetverse."""

__author__ = ["felipeangelimvieira"]  # fkiraly for adapter


import pandas as pd

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.utils.dependencies import _placeholder_record


# TODO 0.40.0: update upper and/or lower bounds when Prophetverse 0.11.0 is released
@_placeholder_record("prophetverse.sktime", dependencies="prophetverse>=0.8.0,<0.11.0")
class Prophetverse(_DelegatedForecaster):
    """Univariate prophetverse forecaster - prophet model implemented in numpyro.

    Estimator from the ``prophetverse`` package by ``felipeangelimvieira``.

    Differences to facebook's prophet:

    * logistic trend. Here, another parametrization is considered,
      and the capacity is not passed as input, but inferred from the data.

    * the users can pass arbitrary ``sktime`` transformers as
      ``feature_transformer``,
      for instance ``FourierFeatures`` or ``HolidayFeatures``.

    * no default weekly_seasonality/yearly_seasonality, this is left to the user
      via the ``feature_transformer`` parameter

    * Uses ``changepoint_interval`` instead of ``n_changepoints`` to set
    changepoints.

    * accepts configurations where each exogenous variable has a different
      function relating it to its additive effect on the time series.
      One can, for example, set different priors for a group of feature,
      or use a Hill function to model the effect of a feature.

    Parameters
    ----------
    trend : Union[str, BaseEffect], optional
        Type of trend to use. Either "linear" (default) or "logistic", or a
        custom effect object.

    exogenous_effects : Optional[List[BaseEffect]], optional
        List of effect objects defining the exogenous effects.

    default_effect : Optional[BaseEffect], optional
        The default effect for variables without a specified effect.

    feature_transformer : sktime transformer, optional
        Transformer object to generate additional features (e.g.,
          Fourier terms).

    noise_scale : float, optional
        Scale parameter for the observation noise. Must be greater than 0.
        (default: 0.05)

    likelihood : str, optional
        The likelihood model to use. One of "normal", "gamma", or
         "negbinomial". (default: "normal")

    scale : optional
        Scaling value inferred from the data.

    rng_key : optional
        A jax.random.PRNGKey instance, or None.

    inference_engine : optional
        An inference engine for running the model.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.prophetverse import Prophetverse
    >>> from prophetverse.effects.fourier import LinearFourierSeasonality
    >>> from prophetverse.utils.regex import no_input_columns
    >>> y = load_airline()
    >>> model = Prophetverse(
    ...     exogenous_effects=[
    ...         (
    ...             "seasonality",
    ...             LinearFourierSeasonality(
    ...                 sp_list=[12],
    ...                 fourier_terms_list=[3],
    ...                 freq="M",
    ...                 effect_mode="multiplicative",
    ...             ),
    ...             no_input_columns,
    ...         )
    ...     ],
    ... )
    >>> model.fit(y)
    >>> model.predict(fh=[1, 2, 3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse>=0.8.0",
        # estimator type
        # --------------
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "enforce_index_type": [pd.Period, pd.DatetimeIndex],
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
        # testing configuration
        # ---------------------
        "tests:vm": True,  # run in VM due to dependency requirement prophetverse
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "_delegate"

    def __init__(
        self,
        trend="linear",
        exogenous_effects=None,
        default_effect=None,
        feature_transformer=None,
        noise_scale=None,
        likelihood="normal",
        scale=None,
        rng_key=None,
        inference_engine=None,
        broadcast_mode="estimator",
    ):
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.trend = trend
        self.exogenous_effects = exogenous_effects
        self.default_effect = default_effect
        self.rng_key = rng_key
        self.scale = scale
        self.likelihood = likelihood
        self.inference_engine = inference_engine
        self.broadcast_mode = broadcast_mode
        super().__init__()

        # delegation, only for prophetverse 0.2.X
        from prophetverse.sktime import Prophetverse

        self._delegate = Prophetverse(**self.get_params())


# TODO 0.41.0: update upper and/or lower bounds when Prophetverse 0.11.0 is released
@_placeholder_record("prophetverse.sktime", dependencies="prophetverse>=0.8.0,<0.11.0")
class HierarchicalProphet(_DelegatedForecaster):
    """A Bayesian hierarchical time series forecasting model based on Meta's Prophet.

    This method forecasts all bottom series in a hierarchy at once, using a
    MultivariateNormal as the likelihood function and LKJ priors for the correlation
    matrix.

    This forecaster is particularly interesting if you want to fit shared coefficients
    across series. In that case, `shared_features` parameter should be a list of
    feature names that should have that behaviour.

    Parameters
    ----------
    trend : Union[str, BaseEffect], optional, default="linear"
        Type of trend to use. Can also be a custom effect object.

    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : Union[float, int], optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * If float, must be between 0 and 1 (inclusive).
          The range will be that proportion of the training history.

        * If int, can be positive or negative.
          Absolute value must be less than the number of training points.
          The range will be that number of points.
          A negative int indicates the number of points
          counting from the end of the history, a positive int from the beginning.

    changepoint_prior_scale : float, optional, default=0.001
        Regularization parameter controlling the flexibility
        of the automatic changepoint selection.

    offset_prior_scale : float, optional, default=0.1
        Scale parameter for the prior distribution of the offset.
        The offset is the constant term in the piecewise trend equation.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    feature_transformer : BaseTransformer or None, optional, default=None
        A transformer to preprocess the exogenous features.

    exogenous_effects : list of AbstractEffect or None, optional, default=None
        A list defining the exogenous effects to be used in the model.

    default_effect : AbstractEffect or None, optional, default=None
        The default effect to be used when no effect is specified for a variable.

    shared_features : list, optional, default=[]
        List of features shared across all series in the hierarchy.

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of warmup steps for MCMC.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains.

    inference_method : str, optional, default='map'
        Inference method to use. Either "map" or "mcmc".

    optimizer_name : str, optional, default='Adam'
        Name of the optimizer to use.

    optimizer_kwargs : dict or None, optional, default={'step_size': 1e-4}
        Additional keyword arguments for the optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps.

    noise_scale : float, optional, default=0.05
        Scale parameter for the noise.

    correlation_matrix_concentration : float, optional, default=1.0
        Concentration parameter for the correlation matrix.

    rng_key : jax.random.PRNGKey or None, optional, default=None
        Random number generator key.

    Examples
    --------
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> from sktime.forecasting.prophetverse import HierarchicalProphet
    >>> agg = Aggregator()
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=3,
    ...     no_levels=1,
    ...     random_seed=123,
    ...     length=7,
    ... )
    >>> y = agg.fit_transform(y)
    >>> forecaster = HierarchicalProphet()
    >>> forecaster.fit(y)
    >>> forecaster.predict(fh=[1])
    """

    _delegate_name = "_delegate"

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse",
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "capability:missing_values": False,
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "fit_is_empty": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,  # run in VM due to dependency requirement prophetverse
    }

    def __init__(
        self,
        trend="linear",
        feature_transformer=None,
        exogenous_effects=None,
        default_effect=None,
        shared_features=None,
        noise_scale=0.05,
        correlation_matrix_concentration=1.0,
        rng_key=None,
        inference_engine=None,
        likelihood=None,
    ):
        self.trend = trend
        self.feature_transformer = feature_transformer
        self.exogenous_effects = exogenous_effects
        self.default_effect = default_effect
        self.shared_features = shared_features
        self.noise_scale = noise_scale
        self.correlation_matrix_concentration = correlation_matrix_concentration
        self.rng_key = rng_key
        self.inference_engine = inference_engine
        self.likelihood = likelihood

        super().__init__()

        # delegation, only for prophetverse 0.2.X
        from prophetverse.sktime import HierarchicalProphet

        self._delegate = HierarchicalProphet(**self.get_params())
