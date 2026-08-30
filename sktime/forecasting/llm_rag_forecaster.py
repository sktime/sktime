"""RAG-enhanced LLM-based forecasting agent for sktime.

This module implements an LLM-based forecasting agent that uses
Retrieval-Augmented Generation (RAG) over sktime's documentation
to make informed forecaster selection decisions.

Unlike simple registry-based approaches, this agent retrieves
relevant documentation sections to understand WHY a forecaster
is suitable for the given data, not just WHICH forecasters exist.
"""

__author__ = ["khushbooshaurya5"]
__all__ = ["RAGForecaster"]

import json
import logging
import warnings
from typing import Optional

import pandas as pd

from sktime.forecasting.base import BaseForecaster

logger = logging.getLogger(__name__)


class RAGForecaster(BaseForecaster):
    """RAG-enhanced LLM-based forecasting agent.

    Uses Retrieval-Augmented Generation (RAG) to select and configure
    an appropriate sktime forecaster based on a natural language query
    and the characteristics of the input time series data.

    The agent retrieves relevant sections from sktime's documentation
    to make informed decisions about forecaster selection, going beyond
    simple registry lookups.

    Parameters
    ----------
    llm_callable : callable
        A callable that takes a single string (prompt) and returns a
        single string (response). Can be a LangChain model, a simple
        function wrapping an API call, or any callable with this
        signature. Example: ``lambda prompt: my_llm.invoke(prompt)``

    query : str, optional, default=None
        Natural language description of the forecasting task.
        Example: "I need a robust model for noisy monthly sales data
        with strong seasonality."
        If None, a default query asking for automatic selection is used.

    doc_paths : list of str, optional, default=None
        Paths to sktime documentation files or directories to use
        as RAG context. If None, uses built-in forecaster docstrings
        from the sktime registry.

    chunk_size : int, optional, default=500
        Number of characters per document chunk for the retrieval step.

    top_k : int, optional, default=3
        Number of top relevant document chunks to retrieve as context.

    fallback_forecaster : str, optional, default="NaiveForecaster"
        Forecaster class name to use if LLM selection fails.

    verbose : bool, optional, default=False
        If True, logs the LLM prompts, retrieved context, and responses.

    Attributes
    ----------
    selected_forecaster_ : BaseForecaster
        The forecaster instance selected and fitted by the agent.

    llm_reasoning_ : str
        The LLM's reasoning for its forecaster selection.

    retrieved_context_ : str
        The document chunks retrieved as context for the LLM.

    Examples
    --------
    >>> from sktime.forecasting.llm_rag_forecaster import RAGForecaster
    >>> from sktime.datasets import load_airline
    >>>
    >>> # Define a simple LLM callable (replace with actual LLM)
    >>> def my_llm(prompt):
    ...     # In practice, use LangChain, OpenAI, etc.
    ...     return '{"forecaster": "NaiveForecaster", "params": {}, '\\
    ...            '"reasoning": "Simple baseline for demonstration."}'
    >>>
    >>> y = load_airline()
    >>> forecaster = RAGForecaster(
    ...     llm_callable=my_llm,
    ...     query="Select a good model for airline passenger data with trend"
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    See Also
    --------
    sktime.registry.all_estimators : List all available estimators.

    Notes
    -----
    The RAG approach retrieves relevant documentation about available
    forecasters, providing the LLM with context about each model's
    strengths, assumptions, and suitable use cases. This leads to
    more informed selections compared to approaches that only pass
    estimator names.

    References
    ----------
    .. [1] Lewis et al., "Retrieval-Augmented Generation for
       Knowledge-Intensive NLP Tasks", NeurIPS 2020.
    """

    _tags = {
        "authors": ["khushbooshaurya5"],
        "maintainers": ["khushbooshaurya5"],
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": None,
    }

    def __init__(
        self,
        llm_callable,
        query=None,
        doc_paths=None,
        chunk_size=500,
        top_k=3,
        fallback_forecaster="NaiveForecaster",
        verbose=False,
    ):
        self.llm_callable = llm_callable
        self.query = query
        self.doc_paths = doc_paths
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.fallback_forecaster = fallback_forecaster
        self.verbose = verbose

        super().__init__()

    def _build_doc_index(self):
        """Build document index from sktime forecaster docstrings.

        Retrieves docstrings from all registered forecasters in sktime
        and splits them into chunks for retrieval.

        Returns
        -------
        chunks : list of dict
            List of document chunks, each with 'text' and 'source' keys.
        """
        from sktime.registry import all_estimators

        chunks = []

        # Get all forecasters from sktime registry
        forecasters = all_estimators(
            estimator_types="forecaster",
            return_names=True,
            as_dataframe=False,
        )

        for name, estimator_class in forecasters:
            docstring = estimator_class.__doc__ or ""
            if not docstring.strip():
                continue

            # Split docstring into chunks
            doc_text = f"Forecaster: {name}\n{docstring}"
            for i in range(0, len(doc_text), self.chunk_size):
                chunk = doc_text[i : i + self.chunk_size]
                chunks.append({"text": chunk, "source": name})

        if self.verbose:
            logger.info(
                f"Built document index with {len(chunks)} chunks "
                f"from {len(forecasters)} forecasters."
            )

        return chunks

    def _retrieve_relevant_chunks(self, query, chunks):
        """Retrieve the most relevant document chunks for a query.

        Uses simple keyword-based similarity for the prototype.
        Can be extended to use vector embeddings (FAISS, Chroma)
        for production use.

        Parameters
        ----------
        query : str
            The user's natural language query.
        chunks : list of dict
            Document chunks from ``_build_doc_index``.

        Returns
        -------
        relevant : list of dict
            Top-k most relevant chunks.
        """
        query_terms = set(query.lower().split())

        scored_chunks = []
        for chunk in chunks:
            chunk_terms = set(chunk["text"].lower().split())
            # Simple term overlap score
            overlap = len(query_terms & chunk_terms)
            scored_chunks.append((overlap, chunk))

        # Sort by score descending and take top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant = [chunk for _, chunk in scored_chunks[: self.top_k]]

        return relevant

    def _extract_data_characteristics(self, y):
        """Extract key characteristics of the time series data.

        Parameters
        ----------
        y : pd.Series
            The time series data.

        Returns
        -------
        characteristics : dict
            Dictionary of data characteristics.
        """
        characteristics = {
            "length": len(y),
            "has_nans": bool(y.isna().any()),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }

        # Try to detect frequency
        if hasattr(y.index, "freq") and y.index.freq is not None:
            characteristics["frequency"] = str(y.index.freq)
        elif hasattr(y.index, "inferred_freq"):
            characteristics["frequency"] = str(
                pd.infer_freq(y.index) or "unknown"
            )
        else:
            characteristics["frequency"] = "unknown"

        # Simple trend detection
        if len(y) > 1:
            first_half_mean = y.iloc[: len(y) // 2].mean()
            second_half_mean = y.iloc[len(y) // 2 :].mean()
            if second_half_mean > first_half_mean * 1.1:
                characteristics["trend"] = "upward"
            elif second_half_mean < first_half_mean * 0.9:
                characteristics["trend"] = "downward"
            else:
                characteristics["trend"] = "stationary"

        return characteristics

    def _build_prompt(self, data_chars, context_chunks):
        """Build the LLM prompt with data characteristics and RAG context.

        Parameters
        ----------
        data_chars : dict
            Data characteristics from ``_extract_data_characteristics``.
        context_chunks : list of dict
            Retrieved document chunks.

        Returns
        -------
        prompt : str
            The formatted prompt for the LLM.
        """
        context_text = "\n\n".join(
            f"--- {chunk['source']} ---\n{chunk['text']}"
            for chunk in context_chunks
        )

        user_query = self.query or (
            "Automatically select the best forecaster for this data."
        )

        prompt = f"""You are an expert time series forecasting advisor.

Based on the following data characteristics and relevant documentation
from the sktime library, select the most appropriate forecaster.

DATA CHARACTERISTICS:
{json.dumps(data_chars, indent=2)}

USER REQUEST:
{user_query}

RELEVANT SKTIME DOCUMENTATION:
{context_text}

INSTRUCTIONS:
1. Analyze the data characteristics (length, frequency, trend, etc.)
2. Consider the user's request
3. Use the documentation context to understand available forecasters
4. Select the most appropriate forecaster

Respond ONLY with a valid JSON object in this exact format:
{{
    "forecaster": "<exact sktime class name>",
    "params": {{}},
    "reasoning": "<brief explanation of why this forecaster was chosen>"
}}

Do not include any text outside the JSON object."""

        return prompt

    def _parse_llm_response(self, response):
        """Parse the LLM response into forecaster name, params, reasoning.

        Parameters
        ----------
        response : str
            Raw LLM response string.

        Returns
        -------
        forecaster_name : str
            Name of the selected forecaster class.
        params : dict
            Hyperparameters for the forecaster.
        reasoning : str
            LLM's reasoning for the selection.
        """
        # Try to extract JSON from response
        try:
            # Handle case where LLM wraps JSON in markdown code blocks
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            parsed = json.loads(cleaned)
            forecaster_name = parsed.get("forecaster", self.fallback_forecaster)
            params = parsed.get("params", {})
            reasoning = parsed.get("reasoning", "No reasoning provided.")
            return forecaster_name, params, reasoning

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            warnings.warn(
                f"Could not parse LLM response: {e}. "
                f"Using fallback forecaster: {self.fallback_forecaster}.",
                UserWarning,
                stacklevel=2,
            )
            return self.fallback_forecaster, {}, f"Fallback due to parse error: {e}"

    def _instantiate_forecaster(self, name, params):
        """Instantiate a forecaster by name from sktime registry.

        Uses sktime's registry for safe instantiation without eval().

        Parameters
        ----------
        name : str
            Forecaster class name.
        params : dict
            Hyperparameters to pass to the constructor.

        Returns
        -------
        forecaster : BaseForecaster
            Instantiated forecaster.
        """
        from sktime.registry import all_estimators

        # Look up forecaster in registry
        forecasters = all_estimators(
            estimator_types="forecaster",
            return_names=True,
            as_dataframe=False,
        )
        forecaster_dict = {name_: cls_ for name_, cls_ in forecasters}

        if name in forecaster_dict:
            try:
                return forecaster_dict[name](**params)
            except Exception as e:
                warnings.warn(
                    f"Could not instantiate {name} with params {params}: {e}. "
                    f"Trying without params.",
                    UserWarning,
                    stacklevel=2,
                )
                try:
                    return forecaster_dict[name]()
                except Exception:
                    pass

        # Fallback
        warnings.warn(
            f"Forecaster '{name}' not found in registry. "
            f"Using {self.fallback_forecaster}.",
            UserWarning,
            stacklevel=2,
        )
        if self.fallback_forecaster in forecaster_dict:
            return forecaster_dict[self.fallback_forecaster]()

        # Ultimate fallback
        from sktime.forecasting.naive import NaiveForecaster

        return NaiveForecaster()

    def _fit(self, y, X=None, fh=None):
        """Fit the RAG forecaster to training data.

        Builds document index, retrieves relevant context,
        queries the LLM for forecaster selection, and fits
        the selected forecaster.

        Parameters
        ----------
        y : pd.Series
            Target time series to forecast.
        X : pd.DataFrame, optional
            Exogenous variables.
        fh : ForecastingHorizon, optional
            Forecasting horizon.

        Returns
        -------
        self : RAGForecaster
            Reference to self.
        """
        # Step 1: Build document index from sktime docs
        chunks = self._build_doc_index()

        # Step 2: Extract data characteristics
        data_chars = self._extract_data_characteristics(y)

        # Step 3: Retrieve relevant documentation
        query_for_retrieval = self.query or "best forecaster for time series"
        # Enrich query with data characteristics for better retrieval
        enriched_query = (
            f"{query_for_retrieval} "
            f"trend:{data_chars.get('trend', '')} "
            f"frequency:{data_chars.get('frequency', '')} "
            f"length:{data_chars.get('length', '')}"
        )
        relevant_chunks = self._retrieve_relevant_chunks(
            enriched_query, chunks
        )
        self.retrieved_context_ = "\n".join(
            c["text"] for c in relevant_chunks
        )

        if self.verbose:
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
            for chunk in relevant_chunks:
                logger.info(f"  Source: {chunk['source']}")

        # Step 4: Build prompt and query LLM
        prompt = self._build_prompt(data_chars, relevant_chunks)

        if self.verbose:
            logger.info(f"LLM Prompt:\n{prompt}")

        response = self.llm_callable(prompt)

        if self.verbose:
            logger.info(f"LLM Response:\n{response}")

        # Step 5: Parse response and instantiate forecaster
        forecaster_name, params, reasoning = self._parse_llm_response(
            response
        )
        self.llm_reasoning_ = reasoning

        if self.verbose:
            logger.info(
                f"Selected: {forecaster_name}, "
                f"Params: {params}, "
                f"Reasoning: {reasoning}"
            )

        # Step 6: Instantiate and fit the selected forecaster
        self.selected_forecaster_ = self._instantiate_forecaster(
            forecaster_name, params
        )
        self.selected_forecaster_.fit(y=y, X=X, fh=fh)

        return self

    def _predict(self, fh, X=None):
        """Generate forecasts using the selected forecaster.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame, optional
            Exogenous variables.

        Returns
        -------
        y_pred : pd.Series
            Forecasted values.
        """
        return self.selected_forecaster_.predict(fh=fh, X=X)

    def summary(self):
        """Return a summary of the LLM's forecaster selection.

        Returns
        -------
        summary : str
            Human-readable summary including the selected forecaster,
            reasoning, and retrieved context sources.
        """
        if not hasattr(self, "selected_forecaster_"):
            return "RAGForecaster has not been fitted yet."

        selected_name = type(self.selected_forecaster_).__name__
        return (
            f"Selected Forecaster: {selected_name}\n"
            f"Reasoning: {self.llm_reasoning_}\n"
            f"Retrieved Context Preview: "
            f"{self.retrieved_context_[:200]}..."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the testing parameter set.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
        """
        # Simple mock LLM for testing
        def mock_llm(prompt):
            return json.dumps(
                {
                    "forecaster": "NaiveForecaster",
                    "params": {},
                    "reasoning": "Mock LLM selected NaiveForecaster for testing.",
                }
            )

        params = {"llm_callable": mock_llm}
        return params
