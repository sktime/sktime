"""LLM-based agentic forecaster for sktime.

Implements AgenticForecaster: uses an LLM to select and configure a sktime
forecaster from natural language task descriptions.
"""

__author__ = ["Nischal1425"]

import json
import re

from sktime.forecasting.base import BaseForecaster


class AgenticForecaster(BaseForecaster):
    """Forecaster that uses an LLM agent to select and configure a sktime forecaster.

    Given a natural language task description and time series data, an LLM selects
    an appropriate sktime forecaster and its hyperparameters. The chosen forecaster
    is then fitted on the data and used for all predictions.

    Supports Anthropic Claude, OpenAI GPT, and a built-in ``"mock"`` backend for
    testing without API keys.

    Parameters
    ----------
    task_description : str, default=""
        Natural language description of the forecasting task, e.g.
        ``"monthly airline passenger data with strong yearly seasonality"``.
        More context leads to better forecaster selection.
    llm_backend : str, default="auto"
        LLM backend to use. One of:

        - ``"anthropic"``: Anthropic Claude API (requires ``anthropic`` package and
          ``ANTHROPIC_API_KEY`` environment variable).
        - ``"openai"``: OpenAI API (requires ``openai`` package and
          ``OPENAI_API_KEY`` environment variable).
        - ``"auto"``: tries ``anthropic`` first, falls back to ``openai``.
        - ``"mock"``: deterministic built-in selection for testing; always picks
          ``NaiveForecaster``. No API key required.

    llm_model : str, default=None
        Model name passed to the LLM backend. ``None`` uses the backend default:
        ``"claude-haiku-4-5-20251001"`` for Anthropic, ``"gpt-4o-mini"`` for OpenAI.
    max_context_length : int, default=30
        Maximum number of recent observations to include in the LLM prompt.

    Attributes
    ----------
    selected_forecaster_ : str
        Name of the forecaster selected by the LLM after ``fit``.
    selected_params_ : dict
        Parameters chosen by the LLM for the selected forecaster.
    forecaster_ : BaseForecaster
        The fitted forecaster instance.

    Examples
    --------
    Using the mock backend (no API key needed):

    >>> from sktime.forecasting.llm_forecaster import AgenticForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AgenticForecaster(
    ...     task_description="monthly airline passenger data",
    ...     llm_backend="mock",
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])
    AgenticForecaster(...)
    >>> y_pred = forecaster.predict()
    >>> forecaster.selected_forecaster_
    'NaiveForecaster'

    Using the Anthropic backend (requires ``ANTHROPIC_API_KEY``):

    >>> forecaster = AgenticForecaster(  # doctest: +SKIP
    ...     task_description="monthly airline data with strong yearly seasonality",
    ...     llm_backend="anthropic",
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "pd.Series",
        "python_dependencies": [],
        "authors": ["Nischal1425"],
        "maintainers": ["Nischal1425"],
    }

    def __init__(
        self,
        task_description="",
        llm_backend="auto",
        llm_model=None,
        max_context_length=30,
    ):
        self.task_description = task_description
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        super().__init__()

    def _fit(self, y, X, fh):
        selection = self._get_llm_selection(y)

        self.selected_forecaster_ = selection.get("forecaster", "NaiveForecaster")
        self.selected_params_ = selection.get("params", {})

        forecaster_cls = self._resolve_forecaster_class(self.selected_forecaster_)
        try:
            self.forecaster_ = forecaster_cls(**self.selected_params_)
        except TypeError:
            self.forecaster_ = forecaster_cls()

        self.forecaster_.fit(y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh, X=X)

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _get_llm_selection(self, y):
        """Return {"forecaster": name, "params": dict} from LLM or mock."""
        if self.llm_backend == "mock":
            return self._mock_selection()

        prompt = self._build_prompt(y)

        if self.llm_backend == "anthropic":
            return self._call_anthropic(prompt)
        if self.llm_backend == "openai":
            return self._call_openai(prompt)
        if self.llm_backend == "auto":
            try:
                return self._call_anthropic(prompt)
            except Exception:
                return self._call_openai(prompt)

        raise ValueError(
            f"Unknown llm_backend: {self.llm_backend!r}. "
            "Use 'anthropic', 'openai', 'auto', or 'mock'."
        )

    def _build_prompt(self, y):
        """Build an LLM prompt from time series metadata and available forecasters."""
        n = len(y)
        freq = getattr(y.index, "freqstr", "unknown")
        recent = y.tail(min(self.max_context_length, n)).tolist()
        mean_val = float(y.mean())
        std_val = float(y.std())

        forecaster_names = self._get_no_dep_forecaster_names()

        task = self.task_description or "general univariate time series forecasting"

        return (
            "You are a time series forecasting expert using the sktime library.\n"
            "Select the most suitable forecaster for the task below.\n\n"
            f"Task: {task}\n\n"
            "Time series properties:\n"
            f"  - Observations: {n}\n"
            f"  - Frequency: {freq}\n"
            f"  - Mean: {mean_val:.4f}, Std dev: {std_val:.4f}\n"
            f"  - Last {len(recent)} values: {recent}\n\n"
            "Available sktime forecasters (no extra dependencies required):\n"
            f"  {', '.join(forecaster_names)}\n\n"
            "Rules:\n"
            "  1. Choose only from the list above.\n"
            "  2. Choose params that are valid constructor arguments for that class.\n"
            "  3. Use {{}} for params if defaults are appropriate.\n\n"
            'Respond with ONLY a JSON object, no explanation:\n'
            '{"forecaster": "<ClassName>", "params": {<param>: <value>, ...}}'
        )

    def _call_anthropic(self, prompt):
        from skbase.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("anthropic", severity="error")
        import anthropic

        model = self.llm_model or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_llm_response(message.content[0].text)

    def _call_openai(self, prompt):
        from skbase.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("openai", severity="error")
        import openai

        model = self.llm_model or "gpt-4o-mini"
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_llm_response(response.choices[0].message.content)

    def _parse_llm_response(self, text):
        """Extract JSON from LLM response; fall back to NaiveForecaster on failure."""
        text = text.strip()
        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "forecaster" in result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return {"forecaster": "NaiveForecaster", "params": {}}

    def _mock_selection(self):
        """Deterministic selection used for testing (no API call)."""
        return {"forecaster": "NaiveForecaster", "params": {"strategy": "last"}}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_no_dep_forecaster_names():
        """Return names of forecasters that need no extra Python dependencies."""
        from sktime.registry import all_estimators

        return [
            name
            for name, cls in all_estimators(
                estimator_types="forecaster", return_names=True
            )
            if not cls.get_class_tags().get("python_dependencies")
        ]

    @staticmethod
    def _resolve_forecaster_class(name):
        """Return the forecaster class for a given name, defaulting to NaiveForecaster."""
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.registry import all_estimators

        lookup = dict(
            all_estimators(estimator_types="forecaster", return_names=True)
        )
        return lookup.get(name, NaiveForecaster)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : list of dict
        """
        return [
            {
                "task_description": "simple univariate forecasting test",
                "llm_backend": "mock",
            }
        ]
