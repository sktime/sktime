"""Probabilistic Intermittent Demand Forecaster."""

from typing import Literal

import jax.nn
import jax.numpy as jnp
import numpy as np
import numpyro.handlers
import pandas as pd
from numpyro.distributions import (
    Bernoulli,
    LogNormal,
    NegativeBinomial2,
    Normal,
    Poisson,
    TransformedDistribution,
    TruncatedNormal,
)
from numpyro.distributions.transforms import (
    AffineTransform,
    RecursiveLinearTransform,
    SoftplusTransform,
)
from prophetverse.engine import BaseInferenceEngine
from prophetverse.sktime.base import BaseBayesianForecaster
from skpro.distributions import Empirical
from xarray import DataArray

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.hurdle_demand._truncated_discrete import TruncatedDiscrete


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

    def _predict_proba(self, marginal=True, **kwargs):
        if self._is_vectorized:
            return self._vectorize_predict_method("_predict_components", **kwargs)

        predictive_samples = self._get_predictive_samples_dict(**kwargs)
        y_hat = predictive_samples["obs"]

        index = kwargs["fh"].to_absolute(self.cutoff).to_numpy()
        as_array = DataArray(y_hat, dims=["sample", "time"], coords={"time": index})

        as_frame = as_array.to_dataframe(self._y_metadata["feature_names"][0])

        return Empirical(as_frame, time_indep=False)

    def _predict(self, fh, X):
        # TODO: this is technically "wrong" since we should use median rather than mean
        predictive_samples = self.predict_components(fh=fh, X=X)
        return predictive_samples["obs"]

    def model(
        self,
        length: int,
        y: jnp.ndarray,
        X: np.ndarray,
        mask: jnp.ndarray,
        oos: int = 0,
        index: np.array = None,
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


class HurdleDemandForecaster(_BaseProbabilisticDemandForecaster):
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
                I_t &\sim \mathcal{B}(p_t), \\
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
        If True, the probability will be modeled as a time series.

    time_varying_demand: bool, default=False
        Whether to use a time varying demand for the Poisson distribution.
        If True, the demand will be modeled as a time series.

    Notes
    -----
    The model is implemented using the `numpyro` library, which allows for inference
    using both MCMC and VI. MCMC can sometimes be a bit tricky to achieve convergence
    with (measured in terms of R-hat), so it is recommended to use dense masses for
    regression parameters should the default inference engine not work well.

    Examples
    --------
    >>> from sktime.forecasting.hurdle_demand import HurdleDemandForecaster
    >>> from prophetverse.engine import MCMCInferenceEngine
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.transformations.series.fourier import FourierFeatures
    >>> from sktime.datasets import load_PBS_dataset
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> numpyro.set_host_device_count(4)
    >>> numpyro.set_platform("cpu")
    >>>
    >>> data = load_PBS_dataset()
    >>> data.index = data.index.to_timestamp() + pd.tseries.offsets.MonthEnd(0)
    >>>
    >>> y_train, y_test = train_test_split(data, test_size=0.3, shuffle=False)
    >>> engine = MCMCInferenceEngine(
    >>>    num_samples=1_000,
    >>>    num_warmup=5_000,
    >>>    num_chains=4,
    >>>    r_hat=1.1,
    >>>    dense_mass=[("probability/beta",), ("demand/beta",)],
    >>> )
    >>> model = HurdleDemandForecaster(
    >>>     time_varying_demand=True,
    >>>     time_varying_probability=True,
    >>>     inference_engine=engine,
    >>> )
    >>> model.fit(y_train)

    See also the notebook under
    `examples/forecasting/probabilistic_intermittent_demand.ipynb`.
    """

    _tags = {
        "authors": ["tingiskhan", "felipeangleimvieira"],
        "maintainers": ["tingiskhan"],
        "python_version": None,
        "python_dependencies": ["prophetverse", "xarray"],
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
        time_varying_probability: bool = False,
        time_varying_demand: bool = False,
        inference_engine: BaseInferenceEngine = None,
    ):
        super().__init__(scale=1.0, inference_engine=inference_engine)

        self.family = family
        self.time_varying_probability = time_varying_probability
        self.time_varying_demand = time_varying_demand

    def _sample_parameters(
        self, length: int, X: np.ndarray, time_regressor: bool = False, oos: int = 0
    ) -> jnp.ndarray:
        features = np.ones((length + oos, 1))

        if X is not None:
            features = np.concatenate((features, X), axis=1)

        with numpyro.plate("factors", features.shape[-1]):
            beta = numpyro.sample("beta", Normal())

        regressors = features @ beta

        if not time_regressor:
            return regressors

        sigma = numpyro.sample("sigma", LogNormal()) ** 0.5
        reversion_speed = numpyro.sample("phi", TruncatedNormal(low=-1.0, high=1.0))

        transition_matrix = reversion_speed.reshape((1, 1))

        eps = Normal().expand((length, 1)).to_event(1)
        time_varying_component = numpyro.sample(
            "x",
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
                "x_oos",
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

    def _sample_probability(
        self, length: int, X: np.ndarray, oos: int = 0
    ) -> jnp.ndarray:
        logit_prob = self._sample_parameters(
            length=length, X=X, time_regressor=self.time_varying_probability, oos=oos
        )
        prob = jax.scipy.special.expit(logit_prob)

        return prob

    def _sample_demand(self, length: int, X: np.ndarray, oos: int = 0) -> jnp.ndarray:
        log_demand = self._sample_parameters(
            length=length, time_regressor=self.time_varying_demand, X=X, oos=oos
        )
        demand = jnp.exp(log_demand)

        return demand

    def model(  # noqa: D102
        self,
        length: int,
        y: jnp.ndarray,
        X: np.ndarray,
        mask: jnp.ndarray,
        oos: int = 0,
        index: np.array = None,
    ):
        with numpyro.handlers.scope(prefix="probability"):
            prob = self._sample_probability(length, X, oos=oos)

        with numpyro.handlers.scope(prefix="demand"):
            demand = self._sample_demand(length, X, oos=oos)

        if index is not None:
            prob = prob[index]
            demand = demand[index]

            events = None
            events_mask = True
            observed_demand = None
        else:
            events = y > 0.0
            events_mask = events
            observed_demand = y

        with numpyro.handlers.mask(mask=mask):
            sampled_events = numpyro.sample(
                "events:ignore", Bernoulli(prob), obs=events
            )

        if self.family == "negative-binomial":
            concentration = numpyro.sample(
                "concentration", TransformedDistribution(Normal(), SoftplusTransform())
            )
            dist = NegativeBinomial2(demand, concentration)
        elif self.family == "poisson":
            dist = Poisson(demand)
        else:
            raise ValueError(f"Unknown family: {self.family}!")

        truncated = TruncatedDiscrete(dist)

        with numpyro.handlers.mask(mask=events_mask & mask):
            sampled_demand = numpyro.sample(
                "demand:ignore", truncated, obs=observed_demand
            )

        if index is None:
            return

        numpyro.deterministic("obs", sampled_events * sampled_demand)

        return
