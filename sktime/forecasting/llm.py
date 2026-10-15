"""LLM-guided forecasting forecaster."""

from __future__ import annotations

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster

__all__ = ["LLMForecaster"]


class _DummyLLM:
    """Simple pickle-safe dummy LLM for testing."""

    def __init__(self, response="FORECASTER: naive\nREASON: stable baseline"):
        self.response = response

    def invoke(self, prompt):
        """Return a fixed response."""
        return self.response


class LLMForecaster(BaseForecaster):
    """LLM-guided sktime forecaster.

    This forecaster uses a user-supplied LLM backend to select a forecasting
    object from a candidate pool, then fits that object and delegates prediction
    to it.

    Parameters
    ----------
    llm : object
        Backend object implementing an ``invoke(prompt: str) -> str`` method.
    candidate_forecasters : tuple of tuple[str, object], optional
        Candidate forecasting objects that the LLM can choose from.
        Entries may be forecaster instances, forecaster classes, or compatible
        forecasting composites/pipelines.
        If None, the full sktime forecaster registry is used.
    default_forecaster : BaseForecaster, optional
        Fallback forecaster used if the LLM output cannot be parsed.
        If None, defaults to ``NaiveForecaster()``.
    prompt_template : str, optional
        Optional custom prompt template.
    strategy : str, default="select"
        Strategy used by the forecaster. Only ``"select"`` is supported.
    """

    _tags = {
        "authors": ["AdithyaPhaniThota"],
        "capability:multivariate": True,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:missing_values": False,
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        llm,
        candidate_forecasters=None,
        default_forecaster=None,
        prompt_template=None,
        strategy="select",
    ):
        self.llm = llm
        self.candidate_forecasters = candidate_forecasters
        self.default_forecaster = default_forecaster
        self.prompt_template = prompt_template
        self.strategy = strategy
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        if self.strategy != "select":
            raise ValueError(
                f"Unsupported strategy={self.strategy!r}. "
                "Only 'select' is supported in v1."
            )

        if not hasattr(self.llm, "invoke"):
            raise TypeError("llm must implement an 'invoke(prompt)' method.")

        candidates = self._get_candidate_forecasters()

        fallback = self.default_forecaster
        if fallback is None:
            fallback = NaiveForecaster()

        self.last_prompt_ = self._build_prompt(y=y, fh=fh, candidates=candidates)
        response = self.llm.invoke(self.last_prompt_)
        self.last_response_ = response

        selected_name, selected_forecaster = self._parse_llm_response(
            response=response,
            candidates=candidates,
        )

        if selected_forecaster is None:
            selected_name = type(fallback).__name__
            selected_forecaster = fallback

        self.selected_forecaster_ = selected_name
        self.forecaster_ = self._coerce_forecaster(selected_forecaster)
        self.forecaster_.fit(y=y, X=X, fh=fh)

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon."""
        return self.forecaster_.predict(fh=fh, X=X)

    def _get_candidate_forecasters(self):
        """Return candidate forecasting objects."""
        if self.candidate_forecasters is not None:
            return self.candidate_forecasters

        from sktime.registry import all_estimators

        forecasters = all_estimators(
            estimator_types="forecaster",
            return_names=True,
        )

        return tuple(
            (name, forecaster)
            for name, forecaster in forecasters
            if name != self.__class__.__name__
        )

    def _coerce_forecaster(self, forecaster):
        """Return a fitted-ready forecaster instance."""
        if isinstance(forecaster, type):
            return forecaster()

        if hasattr(forecaster, "clone"):
            return forecaster.clone()

        raise TypeError(
            "Candidate forecasting objects must be forecaster classes "
            "or cloneable forecaster instances."
        )

    def _build_prompt(self, y, fh, candidates):
        """Build prompt for the LLM."""
        candidate_names = [name for name, _ in candidates]

        length = len(y)
        y_min = float(y.min().min()) if hasattr(y.min(), "min") else float(y.min())
        y_max = float(y.max().max()) if hasattr(y.max(), "max") else float(y.max())
        y_mean = (
            float(y.mean().mean()) if hasattr(y.mean(), "mean") else float(y.mean())
        )
        fh_repr = str(fh)

        if self.prompt_template is not None:
            return self.prompt_template.format(
                length=length,
                y_min=y_min,
                y_max=y_max,
                y_mean=y_mean,
                fh=fh_repr,
                candidates=", ".join(candidate_names),
            )

        candidate_block = "\n".join(f"- {name}" for name in candidate_names)

        return (
            "You are selecting the best forecasting object.\n\n"
            f"Series summary:\n"
            f"- length: {length}\n"
            f"- min: {y_min:.4f}\n"
            f"- max: {y_max:.4f}\n"
            f"- mean: {y_mean:.4f}\n\n"
            f"Forecast horizon:\n{fh_repr}\n\n"
            f"Candidate forecasting objects:\n{candidate_block}\n\n"
            "Return exactly in this format:\n"
            "FORECASTER: <name>\n"
            "REASON: <short explanation>"
        )

    def _parse_llm_response(self, response, candidates):
        """Parse LLM response and return selected candidate."""
        if not isinstance(response, str):
            return None, None

        response_lower = response.lower()

        for name, forecaster in candidates:
            if f"forecaster: {name.lower()}" in response_lower:
                return name, forecaster

        for name, forecaster in candidates:
            if name.lower() in response_lower:
                return name, forecaster

        return None, None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        params1 = {
            "llm": _DummyLLM(),
            "candidate_forecasters": (("naive", NaiveForecaster()),),
            "default_forecaster": NaiveForecaster(),
        }

        params2 = {
            "llm": _DummyLLM("invalid response"),
            "candidate_forecasters": (("naive", NaiveForecaster()),),
            "default_forecaster": NaiveForecaster(),
        }

        return [params1, params2]
