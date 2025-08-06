"""Probabilistic Intermittent Demand Forecaster."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpyro.handlers
import pandas as pd
from jax.scipy.special import expit
from numpyro.distributions import (
    HalfNormal,
    NegativeBinomial2,
    Normal,
    Poisson,
    TransformedDistribution,
)
from numpyro.distributions.transforms import (
    AffineTransform,
    RecursiveLinearTransform,
    SigmoidTransform,
)
from prophetverse.sktime.base import BaseBayesianForecaster
from skpro.distributions import (
    Hurdle as skpro_Hurdle,
)
from skpro.distributions import (
    NegativeBinomial as skpro_NegativeBinomial,
)
from skpro.distributions import Poisson as skpro_Poisson

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.hurdle_demand._truncated_discrete import TruncatedDiscrete

from ._hurdle_distribution import HurdleDistribution


def _sample_components(
    length: int,
    X: np.ndarray,
    mean_reverting: bool,
    use_timeseries: bool = False,
    oos: int = 0,
) -> np.ndarray:
    features = np.ones((length + oos, 1))

    if X is not None:
        features = np.concatenate((features, X), axis=1)

    with numpyro.plate("factors", features.shape[-1]):
        beta = numpyro.sample("beta", Normal())

    regressors = features @ beta

    if not use_timeseries:
        return regressors

    sigma = numpyro.sample("sigma", HalfNormal())

    reversion_speed = 1.0
    if mean_reverting:
        reversion_speed = numpyro.sample(
            "phi", TransformedDistribution(Normal(scale=1.5), SigmoidTransform())
        )

    transition_matrix = jnp.reshape(reversion_speed, (1, 1))

    eps = Normal().expand((length, 1)).to_event(1)
    time_varying_component = numpyro.sample(
        "x:ignore",
        TransformedDistribution(
            eps,
            [
                AffineTransform(0.0, sigma),
                RecursiveLinearTransform(transition_matrix=transition_matrix),
            ],
        ),
    )

    if oos > 0:
        mean = jnp.eye(oos, 1) * time_varying_component[-1] * reversion_speed
        eps_oos = Normal().expand((oos, 1)).to_event(1)

        x_oos = numpyro.sample(
            "x_oos:ignore",
            TransformedDistribution(
                eps_oos,
                [
                    AffineTransform(mean, sigma),
                    RecursiveLinearTransform(transition_matrix=transition_matrix),
                ],
            ),
        )

        time_varying_component = jnp.concatenate(
            (time_varying_component, x_oos), axis=0
        )

    return regressors + time_varying_component.squeeze(-1)


# TODO: think about priors, can we make them more informative?
# TODO: add updating logic based on using means of posterior samples
#  (do this in prophetverse
class _BaseProbabilisticDemandForecaster(BaseBayesianForecaster):
    """Base class for probabilistic intermittent demand forecasters."""

    def _get_fit_data(self, y: pd.DataFrame, X: pd.DataFrame, fh: ForecastingHorizon):
        return {
            "length": y.shape[0],
            "y": y.values,
            "X": X.values if X is not None else None,
            "mask": jnp.isfinite(y.values),
        }

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon):
        if X is not None:
            temp = self._X.copy()
            temp.update(X)

            oos_index = X.index.difference(temp.index)
            if oos_index.size > 0:
                X = pd.concat([temp, X.loc[oos_index]], axis=0)

        index = fh.to_absolute_int(self._y.index[0], self._cutoff)
        oos = fh.to_out_of_sample(self.cutoff).to_numpy().size

        return {
            "length": self._y.shape[0],
            "y": None,
            "X": X.values if X is not None else None,
            "oos": oos,
            "index": index.to_numpy(),
            "mask": True,
        }

    def _get_distribution(
        self, samples: dict[str, np.ndarray], index: pd.DatetimeIndex
    ):
        raise NotImplementedError()

    def _predict_proba(self, marginal=True, **kwargs):
        if self._is_vectorized:
            return self._vectorize_predict_method("_predict_components", **kwargs)

        samples = self._get_predictive_samples_dict(**kwargs)

        index = kwargs["fh"].to_absolute(self.cutoff).to_pandas()
        base_distribution = self._get_distribution(samples, index)

        p = samples["gate"].mean(axis=0)[..., np.newaxis]

        return skpro_Hurdle(p, base_distribution)

    def model(
        self,
        length: int,
        y: np.ndarray,
        X: np.ndarray,
        mask: np.ndarray,
        oos: int = 0,
        index: np.ndarray = None,
    ):
        """
        Build the model for the probabilistic intermittent demand forecaster.

        Parameters
        ----------
        length: int
            Length of the series to sample.
        y: jnp.ndarray
            Observed values.
        X: np.ndarray
            Exogenous variables.
        mask: jnp.ndarray
            Mask for the observed values.
        oos: int
            Number of out-of-sample points to forecast.
        index: np.array
            Index to select.

        Returns
        -------
            Nothing.
        """
        raise NotImplementedError()


class _HurdleDemandForecaster(_BaseProbabilisticDemandForecaster):
    r"""Probabilistic Intermittent Demand Forecaster using a hurdle model.

    The definition of the model is as follows:
        .. math::
            Y_t = \begin{cases}
                D_t \vert r_t \sim f(d \vert r_t) &\text{ if } I_t = 1, \\
                0 &\text{ if } I_t = 0.
            \end{cases}
    where
        .. math::
            \begin{split}
                I_t &\sim \mathcal{B}(1.0 - p_t), \\
                \log{r_t} &= \beta_r \cdot X_t + \Phi(t, r_{t - 1}), \\
                \sigma^{-1}(p_t) &= \beta_p \cdot X_t + \Phi \left ( t,
                \sigma^{-1}(p_t) \right ), \\
                \Phi_i(t, x) &= \phi_i x + \eta_{t, i}, \eta \sim \mathcal{N}(0,
                \sigma^2_i),
            \end{split}
    :math:`f` denotes a density parameterized by at least a location parameter,
    :math:`X` is the exogenous variables, and :math:`\sigma^{-1}` denotes the logit
    function. The time varying component can be toggled on or off depending on the
    value of the `time_varying_<probability|demand>` parameter.

    Parameters
    ----------
    family: str, default="gamma-poisson"
        The family of the model. Can be either "poisson" or "gamma-poisson".

    time_varying_probability: bool, default=False
        Whether to use a time varying probability for the Bernoulli distribution.
        If True, the probability will be modeled as an AR(1) process with positive
        but stationary reversion speed.

    time_varying_demand: bool, default=False
        Whether to use a time varying demand for the Poisson distribution.
        If True, the demand will be modeled as an AR(1) process with positive
        but stationary reversion speed.

    Notes
    -----
    The model is implemented using the `numpyro` library, which allows for inference
    using both MCMC and VI. MCMC can sometimes be a bit tricky to achieve convergence
    with (measured in terms of R-hat), so it is recommended to use dense masses for
    regression parameters should the default inference engine not work well.
    """

    _tags = {
        "authors": ["tingiskhan", "felipeangleimvieira"],
        "maintainers": ["tingiskhan"],
        "python_version": None,
        "python_dependencies": ["prophetverse", "jax", "numpyro", "skpro"],
        "object_type": "forecaster",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "fit_is_empty": False,
        "capability:categorical_in_X": True,
    }

    def __init__(
        self,
        family: Literal["poisson", "negative-binomial"] = "negative-binomial",
        time_varying_probability: Literal["ar", "rw", False] = False,
        time_varying_demand: Literal["ar", "rw", False] = False,
        inference_engine=None,
    ):
        super().__init__(scale=1.0, inference_engine=inference_engine)

        self.family = family
        self.time_varying_probability = time_varying_probability
        self.time_varying_demand = time_varying_demand

    def _sample_probability(
        self, length: int, X: np.ndarray, oos: int = 0
    ) -> np.ndarray:
        use_timeseries = self.time_varying_probability is not False
        mean_reverting = self.time_varying_probability == "ar"

        logit_prob = _sample_components(
            length=length,
            X=X,
            use_timeseries=use_timeseries,
            mean_reverting=mean_reverting,
            oos=oos,
        )
        prob = expit(logit_prob)

        return prob

    def _sample_demand(self, length: int, X: np.ndarray, oos: int = 0) -> np.ndarray:
        use_timeseries = self.time_varying_demand is not False
        mean_reverting = self.time_varying_demand == "ar"

        log_demand = _sample_components(
            length=length,
            use_timeseries=use_timeseries,
            mean_reverting=mean_reverting,
            X=X,
            oos=oos,
        )
        demand = jnp.exp(log_demand)

        return demand

    def _get_distribution(
        self,
        samples,
        index,
    ):
        mu = samples["demand"].mean(axis=0)
        if self.family == "negative-binomial":
            alpha = self.posterior_samples_["concentration"]

            # NB: I had thought numpy would have handled this internally?
            if alpha.size > 1:
                alpha = alpha.mean(axis=0)

            return skpro_NegativeBinomial(mu, alpha, index=index)

        elif self.family == "poisson":
            return skpro_Poisson(mu, index=index)

        raise NotImplementedError(f"Unknown family: {self.family}!")

    def model(  # noqa: D102
        self,
        length: int,
        y: np.ndarray,
        X: np.ndarray,
        mask: np.ndarray,
        oos: int = 0,
        index: np.ndarray = None,
    ):
        with numpyro.handlers.scope(prefix="probability"):
            prob = self._sample_probability(length, X, oos=oos)

        with numpyro.handlers.scope(prefix="demand"):
            demand = self._sample_demand(length, X, oos=oos)

        observed_demand = y
        if index is not None:
            prob = prob[index]
            demand = demand[index]

            observed_demand = None

        if self.family == "negative-binomial":
            concentration = numpyro.sample("concentration", HalfNormal())
            dist = NegativeBinomial2(demand, concentration)
        elif self.family == "poisson":
            dist = Poisson(demand)
        else:
            raise ValueError(f"Unknown family: {self.family}!")

        truncated = TruncatedDiscrete(dist, low=0)

        with numpyro.handlers.mask(mask=mask):
            dist = HurdleDistribution(prob, truncated)
            samples = numpyro.sample("demand:ignore", dist, obs=observed_demand)

        if index is None:
            return

        numpyro.deterministic("gate", prob)
        numpyro.deterministic("demand", demand)
        numpyro.deterministic("obs", samples)

        return

    def predict_components(self, fh, X=None):
        if self._is_vectorized:
            return self._vectorize_predict_method("predict_components", X=X, fh=fh)

        fh_as_index = self.fh_to_index(fh)

        X_inner = self._check_X(X=X)
        predictive_samples_ = self._get_predictive_samples_dict(fh=fh, X=X_inner)

        moment_functions = {
            "gate": np.mean,
            "demand": np.mean,
            "obs": np.median,
        }

        out = pd.DataFrame(
            data={
                site: moment_functions[site](data, axis=0).flatten()
                for site, data in predictive_samples_.items()
            },
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()

        return self._inv_scale_y(out)

    def _predict(self, fh, X):
        predictive_samples = self.predict_components(fh=fh, X=X)
        mean = predictive_samples["obs"]

        col_names = self._y_metadata["feature_names"]
        y_pred = mean.to_frame(col_names[0])

        return self._postprocess_output(y_pred)
