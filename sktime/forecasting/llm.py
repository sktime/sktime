"""Implements LLMForecaster - an LLM-based time series forecasting agent.

Addresses: https://github.com/sktime/sktime/issues/9721
"""

__author__ = ["yashkotha"]
__all__ = ["LLMForecaster"]

import json
import re

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster


class LLMForecaster(BaseForecaster):
    """LLM-based time series forecasting agent.

    Uses a Large Language Model to forecast time series by converting
    historical observations into a structured text prompt, querying the LLM,
    and parsing the structured numeric response.

    Supports any OpenAI-compatible backend, Anthropic, Groq, or any
    ``langchain`` chat model passed directly as ``llm``.

    Parameters
    ----------
    llm : str or langchain chat model, default="openai"
        LLM backend.  Accepted string values:

        * ``"openai"``   – uses :class:`openai.OpenAI` (requires ``openai``).
        * ``"anthropic"`` – uses :class:`anthropic.Anthropic`
          (requires ``anthropic``).
        * ``"groq"``     – uses :class:`groq.Groq` (requires ``groq``).
        * ``"langchain"`` – wraps ``langchain_openai.ChatOpenAI``
          (requires ``langchain_openai``).

        Alternatively pass any object that exposes a ``langchain``-style
        ``.invoke()`` method accepting a list of messages.
    model_name : str or None, default=None
        Model name forwarded to the backend.  Defaults per backend:

        * openai    → ``"gpt-4o-mini"``
        * anthropic → ``"claude-3-5-haiku-20241022"``
        * groq      → ``"llama-3.3-70b-versatile"``
        * langchain → ``"gpt-4o-mini"``
    context_length : int, default=50
        Maximum number of most-recent observations included in the prompt.
    temperature : float, default=0.0
        Sampling temperature.  ``0.0`` yields near-deterministic output.
    verbose : bool, default=False
        If ``True``, prints the prompt sent to the LLM and its raw response.

    Examples
    --------
    Using the OpenAI backend (requires ``OPENAI_API_KEY`` env var):

    >>> from sktime.forecasting.llm import LLMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = LLMForecaster(llm="openai", context_length=24)
    LLMForecaster(...)
    >>> forecaster.fit(y, fh=[1, 2, 3])
    LLMForecaster(...)
    >>> forecaster.predict()  # doctest: +SKIP
    ...

    Using a custom langchain model:

    >>> from langchain_openai import ChatOpenAI  # doctest: +SKIP
    >>> chat = ChatOpenAI(model="gpt-4o", temperature=0)  # doctest: +SKIP
    >>> forecaster = LLMForecaster(llm=chat)  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    LLMForecaster(...)
    """

    _tags = {
        "authors": ["yashkotha"],
        "maintainers": ["yashkotha"],
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "fit_is_empty": False,
        "python_dependencies": None,  # soft deps checked at runtime
    }

    def __init__(
        self,
        llm="openai",
        model_name=None,
        context_length=50,
        temperature=0.0,
        verbose=False,
    ):
        self.llm = llm
        self.model_name = model_name
        self.context_length = context_length
        self.temperature = temperature
        self.verbose = verbose
        super().__init__()

    # ------------------------------------------------------------------
    # Core sktime interface
    # ------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):
        """Fit the LLM forecaster.

        Stores the training series and initialises the LLM client.

        Parameters
        ----------
        y : pd.Series
            Target univariate time series.
        X : ignored
        fh : ForecastingHorizon, optional

        Returns
        -------
        self
        """
        self._y = y
        self._llm_client = self._get_llm_client()
        return self

    def _predict(self, fh, X=None):
        """Produce point forecasts via the LLM.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : ignored

        Returns
        -------
        y_pred : pd.Series
            Predicted values indexed by the absolute forecasting horizon.
        """
        y_context = self._y.iloc[-self.context_length :]
        n_ahead = len(fh)

        prompt = self._build_forecast_prompt(y_context, n_ahead)

        if self.verbose:
            print("=== LLMForecaster prompt ===")
            print(prompt)

        response = self._query_llm(prompt)

        if self.verbose:
            print("=== LLMForecaster response ===")
            print(response)

        predictions = self._parse_predictions(response, n_ahead)
        fh_abs = fh.to_absolute(self.cutoff)
        return pd.Series(predictions, index=fh_abs.to_pandas(), name=self._y.name)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_forecast_prompt(self, y_context, n_ahead):
        """Build the natural-language forecasting prompt.

        Parameters
        ----------
        y_context : pd.Series
            Recent observations to include as context.
        n_ahead : int
            Number of future steps to forecast.

        Returns
        -------
        prompt : str
        """
        values_str = ", ".join(f"{v:.6g}" for v in y_context.values)
        timestamps_str = ", ".join(str(i) for i in y_context.index)

        freq = getattr(y_context.index, "freq", None)
        freq_clause = f" (frequency: {freq})" if freq else ""

        prompt = (
            "You are an expert time series analyst.\n\n"
            f"Historical series{freq_clause}:\n"
            f"  Timestamps : {timestamps_str}\n"
            f"  Values     : {values_str}\n\n"
            f"Task: Forecast the next {n_ahead} value(s) immediately following "
            "the last timestamp shown above.\n\n"
            "Instructions:\n"
            "  1. Identify trend, seasonality, and level from the series.\n"
            "  2. Extrapolate to produce the requested number of future values.\n"
            f"  3. Reply with ONLY a valid JSON object containing key "
            f'"predictions" mapping to a list of exactly {n_ahead} float(s).\n'
            "  4. No explanation, no markdown fences – pure JSON only.\n\n"
            f'Example: {{"predictions": [120.5, 121.0, 119.8]}}\n\n'
            "Your response:"
        )
        return prompt

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _query_llm(self, prompt):
        """Send *prompt* to the LLM and return the raw text response.

        Parameters
        ----------
        prompt : str

        Returns
        -------
        response : str
        """
        client = self._llm_client

        # langchain interface (.invoke returns an AIMessage)
        if hasattr(client, "invoke"):
            from langchain_core.messages import HumanMessage

            msg = client.invoke([HumanMessage(content=prompt)])
            return msg.content

        # openai / groq direct interface
        if hasattr(client, "chat"):
            resp = client.chat.completions.create(
                model=self._model_name_,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content

        # anthropic direct interface
        if hasattr(client, "messages"):
            resp = client.messages.create(
                model=self._model_name_,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        raise TypeError(
            f"Unrecognised LLM client type: {type(client)}.  "
            "Pass a langchain chat model, openai.OpenAI(), anthropic.Anthropic(), "
            "or groq.Groq() instance."
        )

    def _parse_predictions(self, response, n_ahead):
        """Parse *n_ahead* numeric predictions from the LLM response.

        Attempts JSON parsing first; falls back to regex extraction.

        Parameters
        ----------
        response : str
        n_ahead : int

        Returns
        -------
        predictions : list of float

        Raises
        ------
        ValueError
            If fewer than *n_ahead* numeric values can be extracted.
        """
        # --- attempt JSON ---
        try:
            json_match = re.search(r"\{.*?\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                preds = data.get("predictions", data.get("forecast", []))
                if preds and len(preds) >= n_ahead:
                    return [float(v) for v in preds[:n_ahead]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # --- fallback: harvest all numbers ---
        numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", response)
        if len(numbers) >= n_ahead:
            return [float(n) for n in numbers[:n_ahead]]

        raise ValueError(
            f"Could not parse {n_ahead} prediction(s) from LLM response:\n"
            f"{response!r}\n"
            "Ensure the LLM is reachable and returning valid JSON."
        )

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _get_llm_client(self):
        """Instantiate and return the LLM client object.

        Returns
        -------
        client : object
        """
        if not isinstance(self.llm, str):
            # Caller provided a ready-made object (e.g. a langchain model)
            self._model_name_ = getattr(
                self.llm, "model_name", getattr(self.llm, "model", "custom")
            )
            return self.llm

        if self.llm == "openai":
            _check_soft_dependencies("openai", severity="error")
            import openai

            self._model_name_ = self.model_name or "gpt-4o-mini"
            return openai.OpenAI()

        if self.llm == "anthropic":
            _check_soft_dependencies("anthropic", severity="error")
            import anthropic

            self._model_name_ = self.model_name or "claude-3-5-haiku-20241022"
            return anthropic.Anthropic()

        if self.llm == "groq":
            _check_soft_dependencies("groq", severity="error")
            import groq

            self._model_name_ = self.model_name or "llama-3.3-70b-versatile"
            return groq.Groq()

        if self.llm == "langchain":
            _check_soft_dependencies("langchain_openai", severity="error")
            from langchain_openai import ChatOpenAI

            self._model_name_ = self.model_name or "gpt-4o-mini"
            return ChatOpenAI(
                model=self._model_name_, temperature=self.temperature
            )

        raise ValueError(
            f"Unknown llm backend: {self.llm!r}.  "
            "Choose from 'openai', 'anthropic', 'groq', 'langchain', "
            "or pass a langchain chat model instance directly."
        )

    # ------------------------------------------------------------------
    # sktime testing hook
    # ------------------------------------------------------------------

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Uses a :class:`unittest.mock.MagicMock` as the LLM backend so that
        tests run without real API credentials.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict
        """
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            '{"predictions": [100.0, 101.0, 102.0, 103.0, 104.0]}'
        )
        return {"llm": mock_llm, "context_length": 10}
