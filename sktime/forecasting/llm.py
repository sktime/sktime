"""LLM-based forecasting agent.

This module provides the LLMForecaster which acts as an agent translating
user's textual prompts into instantiated sktime forecasting models.
"""

import inspect
import json

from sktime.forecasting.base import BaseForecaster
from sktime.registry import all_estimators

__author__ = ["sktime-contributors"]
__all__ = ["LLMForecaster"]


class LLMForecaster(BaseForecaster):
    """LLM Forecasting Agent that translates text instructions into tasks.

    This forecaster takes a user prompt as input and uses a provided LLM
    backend to select the most appropriate sktime estimator and its
    hyperparameters. The chosen forecaster is then fitted and used for
    prediction.

    Parameters
    ----------
    prompt : str
        The primary text instruction from the user, e.g.,
        "Fit an ARIMA model and forecast the next 12 periods".
    llm : object, default=None
        An LLM backend instance used to process the prompt. To stay agnostic,
        this can accept an object following a standard interface (like a
        LangChain `BaseChatModel`) or a callable that takes a string and
        returns a string response.
    scope : str or list, default="forecaster"
        Defines the scope of the estimators the LLM is allowed to use.
        Currently supports "forecaster".

    Attributes
    ----------
    llm_summary_ : str
        A programmatic text output explaining what the LLM decided to do (e.g.,
        matching the user's intent to ARIMA, parsing hyperparameters).
    estimator_ : sktime estimator
        The internally fitted estimator.

    Examples
    --------
    >>> from sktime.forecasting.llm import LLMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> # Mock LLM for example purposes
    >>> mock_llm = lambda x: (
    ...     '{"estimator": "NaiveForecaster", "params": {}, "explanation": "Naive"}'
    ... )
    >>> agent = LLMForecaster(prompt="Use a Naive model.", llm=mock_llm)
    >>> agent.fit(y)
    LLMForecaster(...)
    >>> print(agent.llm_summary_)
    Naive
    """

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        prompt: str,
        llm=None,
        scope="forecaster",
    ):
        self.prompt = prompt
        self.llm = llm
        self.scope = scope
        super().__init__()

    def _get_available_estimators(self):
        """Retrieve the available estimators based on scope."""
        estimators = all_estimators(estimator_types=self.scope, as_dataframe=False)
        return {name: est for name, est in estimators}

    def _fit(self, y, X=None, fh=None):
        """Fit the chosen forecaster."""
        if self.llm is None:
            raise ValueError("LLM backend `llm` cannot be None.")

        estimators = self._get_available_estimators()
        avail_names = list(estimators.keys())

        system_prompt = (
            "You are a time-series forecasting assistant utilizing "
            "the sktime library.\n"
            f"Available forecasters: {', '.join(avail_names)}.\n"
            "Based on the user's prompt, select the most appropriate sktime forecaster "
            "and suggest appropriate hyper-parameters.\n"
            "Return valid JSON ONLY with the following schema:\n"
            "{\n"
            '  "estimator": "NameOfTheEstimator",\n'
            '  "params": {"param1": "value1", "param2": 12},\n'
            '  "explanation": "Brief explanation of why this was chosen."\n'
            "}"
        )

        # Call the LLM (Langchain duck-typing or plain callable)
        if self.llm == "mock_test_mode":
            content = (
                '{"estimator": "NaiveForecaster", "params": {"strategy": "mean"}, '
                '"explanation": "Test fallback"}'
            )
        elif hasattr(self.llm, "invoke"):
            try:
                from langchain.schema import HumanMessage, SystemMessage
            except ImportError:
                raise ImportError(
                    "langchain is required to use LangChain models. "
                    "Run `pip install langchain`."
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self.prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content
        elif callable(self.llm):
            content = self.llm(system_prompt + "\n\nUser: " + self.prompt)
        else:
            raise TypeError(
                "`llm` must be a Langchain-compatible model (having `.invoke()`), "
                "a callable, or 'mock_test_mode'."
            )

        # Parse output
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            decision = json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM output as JSON:\n{content}") from e

        est_name = decision.get("estimator")
        est_params = decision.get("params", {})
        self.llm_summary_ = decision.get("explanation", "")

        if est_name not in estimators:
            raise ValueError(f"LLM hallucinated an invalid estimator: {est_name}")

        est_class = estimators[est_name]
        valid_params = {}

        sig = inspect.signature(est_class.__init__)
        for k, v in est_params.items():
            if k in sig.parameters:
                valid_params[k] = v

        self.estimator_ = est_class(**valid_params)
        self.clone_tags(self.estimator_)
        self.estimator_.fit(y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None):
        return self.estimator_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        self.estimator_.update(y, X=X, update_params=update_params)
        return self

    def _predict_interval(self, fh, X=None, coverage=None):
        return self.estimator_.predict_interval(fh=fh, X=X, coverage=coverage)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        return self.estimator_.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def _predict_var(self, fh, X=None, cov=False):
        return self.estimator_.predict_var(fh=fh, X=X, cov=cov)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"prompt": "Use NaiveForecaster.", "llm": "mock_test_mode"}]
