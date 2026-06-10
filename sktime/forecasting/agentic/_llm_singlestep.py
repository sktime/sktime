# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements LLM-based agent forecaster for automated model selection."""

__author__ = ["yash-sangwan"]
__all__ = ["LLM1StepAgentForecaster"]

import json

from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.forecasting.base._delegate import _DelegatedForecaster


class LLM1StepAgentForecaster(_DelegatedForecaster):
    r"""Forecaster that uses a single-step LLM or agent to select an sktime model.

    Uses a large language model as an orchestrator to analyze time series
    metadata, select the most appropriate sktime forecasting model, and
    configure its hyperparameters. The LLM or agent does not forecast directly,
    it delegates the actual forecasting to the selected sktime-native model that
    it instantiates and fits on the data.

    The selected model and its configuration are determined during ``fit``,
    in a single step, based on a context containing descriptive metadata about the
    training data and a user-provided query describing the forecasting task or
    constraints. The LLM is prompted to return a JSON object specifying the
    class name of the selected forecaster and a dict of hyperparameters to use
    for that forecaster. The LLM's reasoning for its selection is stored as an
    attribute and can be accessed via the ``user_query_response`` method,
    or the ``user_query_response_`` attribute.

    All subsequent ``predict``, ``update``, and probabilistic forecasting
    calls are delegated to the selected inner forecaster.

    Parameters
    ----------
    llm_backend : callable
        A callable that takes a prompt string and returns a response string.
        The response must contain a JSON block with keys ``"class_name"``,
        ``"kwargs"``, and ``"reasoning"``.

        Example signature: ``def my_llm(prompt: str) -> str``

    query : str
        The user's text constraint or objective for model selection.

        Example: ``"Find a robust model for highly seasonal monthly data"``

    estimators : list of sktime forecasters, or list of (str, forecaster) tuples,
        or None, optional (default=None)
        Allowed forecasters the LLM may select from.
        If None, all forecasters registered via ``sktime.registry.all_estimators``
        are considered.

    prompt_template : str or None, optional (default=None)
        Custom prompt template for the LLM. If None, a default template is used.
        The template may contain the placeholders ``{metadata}``,
        ``{estimators}``, and ``{query}`` which are filled during ``fit``.

        The default template is:

        .. code-block:: text

            "You are an expert time series data scientist. Your task is to select "
            "the single best forecasting model for the dataset described below and "
            "configure its hyperparameters.\n"
            "## User objective\n{query}\n"
            "## Time series metadata\n{metadata}\n"
            "## Available forecasters\n{estimators}\n"
            "## Instructions\n"
            "1. Analyse the metadata and the user objective.\n"
            "2. Select exactly ONE forecaster from the list above.\n"
            "3. Choose appropriate hyperparameters for that forecaster.\n"
            "4. Explain your reasoning briefly.\n"
            "## Required output format\n"
            "Return ONLY a single raw JSON object "
            "(no markdown fences, no commentary outside the JSON). "
            "The JSON must have exactly three keys:\n"
            '{"class_name": "<ClassName>", "kwargs": {...}, "reasoning": "..."}'

    metadata_extactor : callable, optional (default=None)
        Custom function to extract metadata from the training data for the LLM prompt.
        If None, a default extractor is used that gathers descriptive statistics,
        shape information, and other structural properties of the time series.

        Must have signature ``func(y, X, fh) -> dict`` where y, X, fh are arguments
        as passed to ``_fit``, after coercion of ``fh`` to ``ForecastingHorizon``.

        The resulting dict must be json-dumpable, and is passed to the LLM context.

    Attributes
    ----------
    forecaster_ : sktime forecaster
        The forecaster instance selected and fitted by the LLM agent.
        Set during ``fit``.

    user_query_response_ : str
        The LLM's textual reasoning for its model selection, extracted from the
        LLM response during ``fit``.

    See Also
    --------
    MultiplexForecaster : Manual model selection via hyper-parameter tuning.
    TransformSelectForecaster : Rule-based model selection via a transformer.
    FallbackForecaster : Sequential fallback through a list of forecasters.

    Examples
    --------
    >>> from sktime.forecasting.compose import LLMSingleStepAgentForecaster
    >>> from sktime.datasets import load_airline
    >>> import json
    >>> def mock_llm(prompt):
    ...     return json.dumps({
    ...         "class_name": "NaiveForecaster",
    ...         "kwargs": {"strategy": "last"},
    ...         "reasoning": "Simple baseline for demo.",
    ...     })
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = LLMSingleStepAgentForecaster(
    ...     llm_backend=mock_llm,
    ...     query="Pick a simple baseline model",
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    LLMSingleStepAgentForecaster(...)
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        # --- authorship ---
        "authors": ["yash-sangwan"],
        "maintainers": ["yash-sangwan"],
        # --- behavioural: internal types ---
        # pass-through to inner forecaster, no conversion at this level
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        # --- capability tags ---
        # permissive at this level; inner forecaster validates during its own fit
        "requires-fh-in-fit": False,
        "fit_is_empty": False,
        "capability:missing_values": True,
        "capability:exogenous": True,
        "capability:insample": True,
        "capability:multivariate": True,
        # conservative defaults; updated dynamically after inner fit
        # via _set_delegated_tags in _fit
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        # CI and test skip
        # ----------------
        "tests:skip_by_name": [
            "test_predict_time_index_in_sample_full",
            "test_get_fitted_params",  # skipped due to tag handling
            # tags are set in _fit and not in advance due to the opennecess of the
            # LLM selection, which makes them unavailable in __init__
        ],
    }

    # _DelegatedForecaster delegates _predict, _update, _predict_interval, etc.
    # to the attribute named here. Set during _fit.
    _delegate_name = "forecaster_"

    def __init__(
        self,
        llm_backend,
        query,
        estimators=None,
        prompt_template=None,
        metadata_extractor=None,
    ):
        self.llm_backend = llm_backend
        self.query = query
        self.estimators = estimators
        self.prompt_template = prompt_template
        self.metadata_extractor = metadata_extractor

        super().__init__()

    # --- Private helpers --------------------------------------------------

    def _extract_metadata(self, y, X, fh):
        """Extract descriptive metadata from the time series for the LLM prompt.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Training time series.
        X : pd.DataFrame or None
            Exogenous variables.
        fh : ForecastingHorizon or None
            Forecasting horizon.

        Returns
        -------
        metadata : dict
            Dictionary of descriptive statistics and structural information.
        """
        import pandas as pd

        metadata = {}

        # --- shape information ---
        metadata["n_timepoints"] = len(y)
        if isinstance(y, pd.DataFrame):
            metadata["n_variables"] = y.shape[1]
            metadata["variable_names"] = list(y.columns)
        else:
            metadata["n_variables"] = 1

        # --- index / frequency ---
        metadata["index_type"] = type(y.index).__name__
        freq = getattr(y.index, "freq", None)
        if freq is not None:
            metadata["frequency"] = str(freq)
        else:
            freq_inferred = getattr(y.index, "inferred_freq", None)
            metadata["frequency"] = str(freq_inferred) if freq_inferred else "unknown"

        # --- missing values ---
        if isinstance(y, pd.DataFrame):
            metadata["has_nans"] = bool(y.isna().any().any())
        else:
            metadata["has_nans"] = bool(y.isna().any())

        # --- summary statistics ---
        if isinstance(y, pd.DataFrame):
            desc = y.describe().to_dict()
        else:
            desc = y.describe().to_dict()
        # keep only JSON-serialisable scalar values
        metadata["summary_stats"] = {
            k: round(float(v), 6) if isinstance(v, (int, float)) else v
            for k, v in desc.items()
        }

        # --- exogenous variables ---
        metadata["has_exogenous"] = X is not None
        if X is not None:
            metadata["n_exogenous_variables"] = (
                X.shape[1] if isinstance(X, pd.DataFrame) else 1
            )

        # --- forecasting horizon ---
        if fh is not None:
            metadata["fh_length"] = len(fh)
            metadata["fh_is_relative"] = fh.is_relative

        return metadata

    def _get_estimator_scope(self):
        """Build the list of candidate estimator names and descriptions.

        Returns
        -------
        scope : list of dict
            Each dict has keys ``"name"`` (str) and ``"description"`` (str).
        registry : dict
            Mapping of class name (str) to class object, used for safe
            instantiation later.
        """
        scope = []
        registry = {}

        if self.estimators is None:
            from sktime.registry import all_estimators

            est_list = all_estimators(estimator_types="forecaster")
            for name, klass in est_list:
                doc = (klass.__doc__ or "").strip()
                first_line = doc.split("\n")[0] if doc else "No description."
                scope.append({"name": name, "description": first_line})
                registry[name] = klass
        else:
            for entry in self.estimators:
                # handle (name, estimator) tuples or bare estimator instances
                if isinstance(entry, tuple) and len(entry) == 2:
                    name, est = entry
                else:
                    est = entry
                    name = type(est).__name__

                klass = est if isinstance(est, type) else type(est)
                doc = (klass.__doc__ or "").strip()
                first_line = doc.split("\n")[0] if doc else "No description."
                scope.append({"name": name, "description": first_line})
                registry[name] = klass

        return scope, registry

    _DEFAULT_PROMPT_TEMPLATE = (
        "You are an expert time series data scientist. Your task is to select "
        "the single best forecasting model for the dataset described below and "
        "configure its hyperparameters.\n\n"
        "## User objective\n{query}\n\n"
        "## Time series metadata\n{metadata}\n\n"
        "## Available forecasters\n{estimators}\n\n"
        "## Instructions\n"
        "1. Analyse the metadata and the user objective.\n"
        "2. Select exactly ONE forecaster from the list above.\n"
        "3. Choose appropriate hyperparameters for that forecaster.\n"
        "4. Explain your reasoning briefly.\n\n"
        "## Required output format\n"
        "Return ONLY a single raw JSON object (no markdown fences, no "
        "commentary outside the JSON). The JSON must have exactly three keys:\n"
        '{{"class_name": "<ClassName>", "kwargs": {{...}}, "reasoning": "..."}}\n'
    )

    # --- Core methods -----------------------------------------------------

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Uses the LLM backend to select and configure an inner forecaster,
        then fits the selected forecaster on the training data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Training time series.
        X : pd.DataFrame or None
            Exogenous variables.
        fh : ForecastingHorizon or None
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        import re

        # --- context gathering ---
        if self.metadata_extractor is not None:
            metadata = self.metadata_extractor(y, X, fh)
        else:
            metadata = self._extract_metadata(y, X, fh)

        estimator_scope, self._estimator_registry = self._get_estimator_scope()

        # --- prompt construction ---
        template = self.prompt_template or self._DEFAULT_PROMPT_TEMPLATE

        estimator_listing = "\n".join(
            f"- {e['name']}: {e['description']}" for e in estimator_scope
        )

        prompt = template.format(
            query=self.query,
            metadata=json.dumps(metadata, indent=2),
            estimators=estimator_listing,
        )

        # --- LLM call ---
        response_str = self.llm_backend(prompt)

        # --- parse response ---
        # strip markdown code fences if the LLM wrapped the output
        cleaned = re.sub(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            r"\1",
            response_str,
            flags=re.DOTALL,
        )
        # extract the first JSON object in case of surrounding text
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise ValueError(
                "LLM response did not contain a valid JSON object. "
                f"Raw response:\n{response_str}"
            )

        try:
            spec = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM response contained malformed JSON. "
                f"Extracted text:\n{match.group()}"
            ) from exc

        # --- validate spec keys ---
        for key in ("class_name", "kwargs", "reasoning"):
            if key not in spec:
                raise ValueError(
                    f"LLM response JSON is missing required key '{key}'. "
                    f"Got keys: {list(spec.keys())}"
                )

        class_name = spec["class_name"]
        kwargs = spec["kwargs"]
        reasoning = spec["reasoning"]

        if not isinstance(kwargs, dict):
            raise TypeError(
                f"'kwargs' must be a dictionary, got {type(kwargs).__name__}."
            )

        # --- safe instantiation via registry lookup ---
        if class_name not in self._estimator_registry:
            valid = sorted(self._estimator_registry.keys())
            raise ValueError(
                f"LLM selected forecaster '{class_name}' which is not in the "
                f"available estimator scope. Valid forecaster names are:\n{valid}"
            )

        cls = self._estimator_registry[class_name]
        inner_forecaster = cls(**kwargs)

        # --- delegated fit ---
        inner_forecaster.fit(y=y, X=X, fh=fh)
        self.forecaster_ = inner_forecaster

        # --- propagate inner forecaster tags ---
        self._set_delegated_tags(self.forecaster_)

        # --- store LLM reasoning ---
        self.user_query_response_ = reasoning

        return self

    # _predict, _update — inherited from _DelegatedForecaster (delegate to
    #   self.forecaster_ which is set during _fit).

    def user_query_response(self):
        """Return the LLM's reasoning for its model selection.

        This method provides access to the textual explanation the LLM
        generated when selecting and configuring the inner forecaster.

        State required:
            Requires state to be "fitted", i.e., ``fit`` must have been
            called before this method.

        Returns
        -------
        reasoning : str
            The LLM's explanation of why it chose the selected forecaster
            and hyperparameters.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.
        """
        self.check_is_fitted()
        return self.user_query_response_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """

        def _mock_llm_backend(prompt):
            return json.dumps(
                {
                    "class_name": "NaiveForecaster",
                    "kwargs": {"strategy": "last"},
                    "reasoning": "Mock selection for testing.",
                }
            )

        params1 = {
            "llm_backend": _mock_llm_backend,
            "query": "Select any simple forecaster for testing.",
        }

        def _mock_llm_backend_mean(prompt):
            return json.dumps(
                {
                    "class_name": "NaiveForecaster",
                    "kwargs": {"strategy": "mean"},
                    "reasoning": "Mock mean selection for testing.",
                }
            )

        params2 = {
            "llm_backend": _mock_llm_backend_mean,
            "query": "Select a mean-based forecaster.",
            "estimators": None,
        }

        return [params1, params2]
