"""Probabilistic Intermittent Demand Forecaster.

This module implements a probabilistic intermittent demand forecaster using a
Bayesian approach.
"""

import jax.nn
import jax.numpy as jnp
import numpy as np
import numpyro.handlers
import pandas as pd
from numpyro.distributions import Laplace, Normal, ZeroInflatedPoisson
from prophetverse.sktime.base import BaseBayesianForecaster

from sktime.forecasting.base import ForecastingHorizon


# TODO: move these to methods instead to allow for more flexibility in the model
#  structure
def _sample_gate(time_varying: bool, length: int, X: np.ndarray) -> jnp.ndarray:
    """Sample the gate parameter for the ZIP model."""
    regressors = 0.0
    if X is not None:
        beta = numpyro.sample("beta", Laplace(), sample_shape=X.shape[-1:])
        regressors = X @ beta

    if not time_varying:
        logit_gate = numpyro.sample("logit_gate", Normal())
        gate = jax.nn.sigmoid(logit_gate + regressors)

        return jnp.full((length,), gate)

    raise NotImplementedError("Time-varying gate parameters are not implemented yet.")


def _sample_rate(time_varying: bool, length: int, X: np.ndarray) -> jnp.ndarray:
    """Sample the log_rate parameter for the ZIP model."""
    regressors = 0.0
    if X is not None:
        beta = numpyro.sample("beta", Laplace(), sample_shape=X.shape[-1:])
        regressors = X @ beta

    if not time_varying:
        # TODO: consider using a more informative prior
        # TODO: consider using a Gamma prior instead of Exponential
        # TODO: how to handle regressors?
        log_rate = numpyro.sample("log_rate", Normal(scale=10.0))
        rate = jnp.exp(log_rate + regressors)

        return jnp.full((length,), rate)

    raise NotImplementedError(
        "Time-varying log_rate parameters are not implemented yet."
    )


class ProbabilisticIntermittentDemandForecaster(BaseBayesianForecaster):
    r"""Probabilistic Intermittent Demand Forecaster.

    Uses a Bayesian approach to forecast intermittent demand time series by modeling
    the series as an Zero-Inflated Poisson (ZIP) process. The mathematical model is
    given by
    .. math::
        y_t \\sim ZIP(g_t, r_t)

    where :math:`g_t` is the gate parameter and :math:`r_t` is the rate parameter. The
    gate parameter determines the probability of observing a non-zero value, while the
    rate parameter determines the expected value of the non-zero observations. The
    rates and gates can be time-varying or constant, depending on the model
    configuration. The general model structure is as follows:
    .. math::
        \\logit{g_t} = \text{logit\\_gate\\_offset} + \beta_{g}^T X_t,
        \\log{r_t} = \text{log\\_rate\\_offset}, + \beta_{r}^T X_t

    where :math:`X_t` are the exogenous variables, :math:`\beta` are the regression
    coefficients, and TODO.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "TODO",
        "maintainers": "TODO",
        "python_version": None,
        "python_dependencies": ["prophetverse"],
        # estimator type
        # --------------
        "object_type": "forecaster",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": False,  # TODO: consider this
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
        time_varying_gate: bool = False,
        time_varying_rate: bool = False,
        inference_engine=None,
    ):
        super().__init__(scale=1.0, inference_engine=inference_engine)
        self.time_varying_gate = time_varying_gate
        self.time_varying_rate = time_varying_rate

    def _get_fit_data(self, y: pd.DataFrame, X: pd.DataFrame, fh: ForecastingHorizon):
        return {
            "length": y.shape[0],
            "y": jnp.array(y.values),
            "X": jnp.array(X.values) if X is not None else None,
        }

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon):
        if X is not None:
            X = pd.concat([self._X, X], axis=0, verify_integrity=True)

        index = fh.to_absolute_int(self._y.index[0], self._cutoff)
        oos = fh.to_out_of_sample(self.cutoff).to_numpy().size

        return {
            "length": self._y.shape[0] + oos,
            "y": None,
            "X": jnp.array(X.values) if X is not None else None,
            "oos": oos,
            "index": index.to_numpy(),
        }

    def model(
        self,
        length: int,
        y: jnp.ndarray,
        X: np.ndarray,
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
        oos: int
            Number of out-of-sample points to forecast.
        index: np.array
            Index to select.

        Returns
        -------
            Nothing.
        """
        # gate
        with numpyro.handlers.scope(prefix="gate"):
            gate = _sample_gate(self.time_varying_gate, length, X)

        # rate
        with numpyro.handlers.scope(prefix="rate"):
            rate = _sample_rate(self.time_varying_rate, length, X)

        # observed data
        if index is not None:
            gate = gate[index]
            rate = rate[index]

        sampled_y = numpyro.sample(
            "y:ignore", ZeroInflatedPoisson(1.0 - gate, rate), obs=y
        )

        if oos == 0:
            return

        numpyro.deterministic("observed", sampled_y)

        return

    def _predict(self, fh, X):
        predictive_samples = self.predict_components(fh=fh, X=X)
        return predictive_samples["observed"]

    def predict_components(self, fh: ForecastingHorizon, X: pd.DataFrame = None):  # noqa: D102
        if self._is_vectorized:
            return self._vectorize_predict_method("predict_components", X=X, fh=fh)

        fh_as_index = self.fh_to_index(fh)

        X_inner = self._check_X(X=X)
        predictive_samples_ = self._get_predictive_samples_dict(fh=fh, X=X_inner)

        out = pd.DataFrame(
            data={
                site: np.median(data, axis=0).flatten()
                for site, data in predictive_samples_.items()
            },
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()

        return self._inv_scale_y(out)
