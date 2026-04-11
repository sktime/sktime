"""LLM-guided forecasting forecaster."""

from __future__ import annotations

from skbase.base import BaseObject

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster

__all__ = ["LLMForecaster"]


class DummyLLM(BaseObject):
    """Simple clone-safe and pickle-safe dummy LLM for testing."""

    def __init__(self, response="FORECASTER: naive\nREASON: stable baseline"):
        self.response = response
        super().__init__()

    def invoke(self, prompt):
        """Return a fixed response."""
        return self.response


class LLMForecaster(BaseForecaster):
    """LLM-guided sktime forecaster."""

    _tags = {
        "authors": ["AdithyaPhaniThota"],
        "scitype:y": "univariate",
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

        candidates = self.candidate_forecasters
        if candidates is None:
            candidates = (("naive", NaiveForecaster()),)

        fallback = self.default_forecaster
        if fallback is None:
            fallback = NaiveForecaster()

        self.last_prompt_ = self._build_prompt(y=y, fh=fh, candidates=candidates)
        response = self.llm.invoke(self.last_prompt_)
        self.last_response_ = response

        selected_name, selected_estimator = self._parse_llm_response(
            response=response,
            candidates=candidates,
        )

        if selected_estimator is None:
            selected_name = type(fallback).__name__
            selected_estimator = fallback

        self.selected_forecaster_ = selected_name
        self.forecaster_ = selected_estimator.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon."""
        return self.forecaster_.predict(fh=fh, X=X)

    def _build_prompt(self, y, fh, candidates):
        """Build prompt for the LLM."""
        candidate_names = [name for name, _ in candidates]

        length = len(y)
        y_min = float(y.min())
        y_max = float(y.max())
        y_mean = float(y.mean())
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
            "You are selecting the best forecasting model.\n\n"
            f"Series summary:\n"
            f"- length: {length}\n"
            f"- min: {y_min:.4f}\n"
            f"- max: {y_max:.4f}\n"
            f"- mean: {y_mean:.4f}\n\n"
            f"Forecast horizon:\n{fh_repr}\n\n"
            f"Candidate forecasters:\n{candidate_block}\n\n"
            "Return exactly in this format:\n"
            "FORECASTER: <name>\n"
            "REASON: <short explanation>"
        )

    def _parse_llm_response(self, response, candidates):
        """Parse LLM response and return selected candidate."""
        if not isinstance(response, str):
            return None, None

        response_lower = response.lower()

        for name, estimator in candidates:
            if f"forecaster: {name.lower()}" in response_lower:
                return name, estimator

        for name, estimator in candidates:
            if name.lower() in response_lower:
                return name, estimator

        return None, None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {
            "llm": DummyLLM(),
            "candidate_forecasters": (("naive", NaiveForecaster()),),
            "default_forecaster": NaiveForecaster(),
        }
