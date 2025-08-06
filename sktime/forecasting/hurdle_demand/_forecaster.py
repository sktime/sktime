from typing import Literal

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.utils.dependencies import _placeholder_record


# TODO 0.39.0: update upper and lower bounds when Prophetverse 0.9.0 is released
@_placeholder_record("prophetverse.sktime", dependencies="prophetverse>=0.3.0,<0.9.0")
class HurdleDemandForecaster(_DelegatedForecaster):
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
    >>> import numpyro
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
    >>>     time_varying_demand="ar",
    >>>     time_varying_probability="rw",
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
        "python_dependencies": ["prophetverse", "skpro"],
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

    _delegate_name = "_delegate"

    def __init__(
        self,
        family: Literal["poisson", "negative-binomial"] = "negative-binomial",
        time_varying_probability: Literal["ar", "rw", False] = False,
        time_varying_demand: Literal["ar", "rw", False] = False,
        inference_engine=None,
    ):
        self.family = family
        self.time_varying_probability = time_varying_probability
        self.time_varying_demand = time_varying_demand
        self.inference_engine = inference_engine

        super().__init__()

        from ._model import _HurdleDemandForecaster

        self._delegate = _HurdleDemandForecaster(**self.get_params(deep=False))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params_1 = {
            "family": "negative-binomial",
            "time_varying_probability": "rw",
            "time_varying_demand": "rw",
            "inference_engine": None,
        }

        params_2 = {
            "family": "poisson",
            "time_varying_probability": False,
            "time_varying_demand": False,
            "inference_engine": None,
        }

        return params_1, params_2
