"""LLM-driven blueprint generation forecaster for sktime.

Inspired by Karpathy's autoresearch: uses an LLM to iteratively generate,
evaluate, and refine sktime forecasting pipeline blueprints to find the best
combination of transformers and forecasters for a given dataset.
"""

import base64
import io
import json
import warnings
from time import sleep

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.registry import craft

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert time-series forecasting engineer.
Your task is to propose sktime forecasting pipeline blueprints.

A blueprint is a JSON object with:
- "name": a short descriptive name
- "spec": a Python expression passed to craft() to construct the forecaster.
  Use sktime class names directly — no imports needed. craft() resolves them.

Rules for "spec":
- A bare forecaster: NaiveForecaster(strategy="last")
- A pipeline: TransformedTargetForecaster([Transformer1(), ..., Forecaster()])
  The last element must be a forecaster; all preceding elements must be transformers.
- You MUST only use class names from the lists below — any other name will fail.
- Constructor arguments must be JSON-serializable Python literals.
- Be creative — try detrending, deseasonalizing, differencing, Box-Cox, ARIMA, ETS, etc.

Available forecaster class names:
{forecaster_names}

Available transformer class names:
{transformer_names}

Few-shot examples of valid blueprints:
[
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "Naive last",
"spec": "NaiveForecaster(strategy=\\"last\\")",
}},
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "ETS additive",
"spec": "ExponentialSmoothing(trend=\\"add\\", seasonal=\\"add\\", sp=12)",
}},
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "Detrend + ETS",
"spec": "TransformedTargetForecaster([Detrender(), ExponentialSmoothing()])",
}},
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "Deseason + Detrend + AutoARIMA",
"spec": "TransformedTargetForecaster([Deseasonalizer(sp=12),Detrender(), AutoARIMA()])",
}},
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "BoxCox + Naive mean",
"spec": "TransformedTargetForecaster([Detrender(), NaiveForecaster(strategy='mean')])",
}},
{{
"reason": <Reason why you selected this  blueprint for this dataset.>
"name": "Differencer + AutoARIMA BIC",
"spec": "TransformedTargetForecaster([Differencer(), AutoARIMA()])",
}}
]

Provide {n_blueprints} diverse blueprints in a JSON array.
Respond ONLY with a valid JSON array of blueprint objects — no markdown, no explanation.
"""

_FIX_BLUEPRINT_PROMPT = """\
The following blueprint failed with an error:

Name: {name}
Spec: {spec}
Error: {error}

Documentation for the estimators used in this blueprint:
{estimator_docs}

Fix the blueprint so it runs without error. Keep the intent as close as possible.
Respond ONLY with a valid JSON array containing exactly one blueprint object
— no markdown, no explanation.
"""

_REFINEMENT_PROMPT = """\
Here are the results from the previous round of blueprint evaluation:

{results_summary}

All blueprints evaluated so far (across all iterations), ranked by score:
{all_results_ranked}

Based on these results:
1. The best performing blueprint was: {best_name} (score: {best_score:.6f})
2. Estimators already tried and their best scores are listed above — avoid repeating
   exact combinations that failed; prefer unexplored estimators or new configurations.
3. Try to improve on the best blueprint by making targeted modifications.
4. Try to fix any blueprints that failed with errors by using the error messages
   as clues. Append "FIXED" to the name of the blueprint your are trying to fix.
5. Also explore new diverse combinations that might outperform it.

Propose {n_blueprints} new blueprints. At least one should be a variation of the
best blueprint, and at least one should be a novel combination.

Respond ONLY with a valid JSON array of blueprint objects — no markdown, no explanation.
"""


def _get_estimator_docs_from_spec(spec):
    """Extract class names from a blueprint spec and return their docstrings.

    Parses the spec string with a simple regex to find sktime class names,
    looks each one up in the registry, and returns a formatted string of
    name + docstring pairs.  Unknown names are silently skipped.

    Parameters
    ----------
    spec : str
        A craft()-compatible spec expression, e.g.
        "TransformedTargetForecaster([Detrender(), AutoARIMA()])".

    Returns
    -------
    docs : str
        Formatted documentation for each recognised estimator class.
    """
    import inspect
    import re

    from sktime.registry import all_estimators

    class_names = re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", spec)
    registry = dict(
        all_estimators(estimator_types="forecaster")
        + all_estimators(estimator_types="transformer")
    )
    seen = set()
    parts = []
    for name in class_names:
        if name in seen or name not in registry:
            continue
        seen.add(name)
        doc = inspect.getdoc(registry[name]) or "(no docstring)"
        parts.append(f"### {name}\n{doc}")
    return "\n\n".join(parts) if parts else "(no documentation found)"


def _build_estimator_names(info="names"):
    """Return formatted lists of sktime forecasters and transformers.

    Parameters
    ----------
    info : str, default="names"
        Level of detail to include per estimator:
        - "names"      : comma-separated class names only
        - "signatures" : name plus constructor parameter names
        - "docstrings"  : name plus full class docstring
    """
    import inspect

    from sktime.registry import all_estimators

    def _format(estimators):
        if info == "names":
            return ", ".join(name for name, _ in estimators)
        parts = []
        for name, cls in estimators:
            if info == "signatures":
                try:
                    sig = inspect.signature(cls.__init__)
                    params = ", ".join(p for p in sig.parameters if p != "self")
                    parts.append(f"{name}({params})")
                except (ValueError, TypeError):
                    parts.append(name)
            elif info == "docstrings":
                doc = inspect.getdoc(cls) or ""
                parts.append(f"{name}:\n{doc}")
        return "\n".join(parts) if info != "names" else ", ".join(parts)

    forecaster_names = _format(all_estimators(estimator_types="forecaster"))
    transformer_names = _format(all_estimators(estimator_types="transformer"))
    return forecaster_names, transformer_names


def _evaluate_blueprint(blueprint, cv, y, X=None):
    """Evaluate a single blueprint on the given train/test split.

    Returns
    -------
    result : dict
        Dictionary with keys: name, score, error, forecaster, blueprint.
    """
    blueprint_name = blueprint.get("name", "unnamed")
    blueprint_spec = blueprint.get("spec", "")
    try:
        forecaster = craft(blueprint_spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = evaluate(forecaster=forecaster, y=y, X=X, cv=cv)
        score = results["test_MeanAbsolutePercentageError"].mean()
        return {
            "name": blueprint_name,
            "score": float(score),
            "error": None,
            "forecaster": forecaster,
            "blueprint": blueprint,
        }
    except Exception as e:
        return {
            "name": blueprint_name,
            "score": float("inf"),
            "error": f"{type(e).__name__}: {e}",
            "forecaster": None,
            "blueprint": blueprint,
        }


def _call_llm(messages, model, api_params):
    """Call the LLM via litellm and return the response text."""
    import litellm
    from pydantic import BaseModel

    class Blueprint(BaseModel):
        name: str
        spec: str
        reason: str

    class LLMformat(BaseModel):
        blueprints: list[Blueprint]

    litellm.suppress_debug_info = True
    for i in range(5):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=LLMformat,
                **api_params,
            )
            return response.choices[0].message.content
        except litellm.RateLimitError:
            sleep(10)  # simple retry logic for rate limits
    raise RuntimeError("LLM call failed after 5 attempts due to rate limits.")


def _parse_blueprints(response_text):
    """Parse LLM response text into a list of blueprint dicts."""
    return json.loads(response_text)["blueprints"]


def _mock_llm(_messages, _model, _api_params):
    """Mock LLM callable for testing — returns a single naive blueprint."""
    return json.dumps(
        {
            "blueprints": [
                {
                    "name": "Naive last",
                    "spec": 'NaiveForecaster(strategy="last")',
                    "reason": "A simple baseline that often performs decently.",
                }
            ]
        }
    )


def _format_result(result, *, prefix="- "):
    """Format a single result dict as a human-readable status line."""
    if result["error"]:
        return f"{prefix}{result['name']}: FAILED ({result['error']})"
    return f"{prefix}{result['name']}: Score={result['score']:.6f}"


def _build_iteration_summary(iteration_results):
    """Build a formatted summary string for a list of iteration results."""
    return "\n".join(_format_result(r) for r in iteration_results)


def _get_basic_description(y):
    """Generate basic text-only dataset description.

    Parameters
    ----------
    y : pd.Series
        Time series data.

    Returns
    -------
    description : str
        Text description of the dataset.
    """
    return (
        f"Dataset info:\n"
        f"- Length: {len(y)} observations\n"
        f"""- Frequency: {
            y.index.freqstr if hasattr(y.index, "freqstr") else "unknown"
        }\n"""
        f"- Mean: {y.mean():.4f}, Std: {y.std():.4f}\n"
        f"- Min: {y.min():.4f}, Max: {y.max():.4f}\n"
    )


def _generate_time_series_plot_base64(y):
    """Generate a time series plot and encode as base64 PNG.

    Parameters
    ----------
    y : pd.Series
        Time series data.

    Returns
    -------
    base64_img : str
        Base64-encoded PNG image.
    """
    from skbase.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib", obj="AutoResearchForecaster plotting")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.values, linewidth=1.5)
    ax.set_title("Time Series Data")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    # Save to in-memory buffer and encode as base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
    buffer.seek(0)
    base64_img = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return base64_img


def _get_described_plot_description(y, model, api_params):
    """Generate dataset description with VLM-described plot.

    Generates a time series plot, uses a vision LLM to describe it,
    and combines the description with basic statistics.

    Parameters
    ----------
    y : pd.Series
        Time series data.
    model : str
        LLM model identifier compatible with litellm.
    api_params : dict
        Additional parameters for litellm.completion.

    Returns
    -------
    description : str
        Combined text description with plot analysis.
    """
    import litellm

    # Generate plot as base64
    base64_img = _generate_time_series_plot_base64(y)

    # Use VLM to describe the plot
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the key characteristics of this time"
                    "series plot briefly, focusing on trends,"
                    "seasonality, and volatility.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                },
            ],
        }
    ]

    try:
        litellm.suppress_debug_info = True
        response = litellm.completion(
            model=model,
            messages=messages,
            **api_params,
        )
        plot_description = response.choices[0].message.content
    except Exception as e:
        plot_description = f"[Failed to analyze plot: {e}]"

    # Combine basic stats with VLM description
    basic_desc = _get_basic_description(y)
    return f"{basic_desc}\nPlot Analysis:\n{plot_description}\n"


def _get_image_description(y):
    """Generate time series plot as base64 for VLM input (image only).

    Parameters
    ----------
    y : pd.Series
        Time series data.

    Returns
    -------
    description : str
        Basic dataset info.
    base64_img : str
        Base64-encoded PNG image for use in blueprint generation.
    """
    basic_desc = _get_basic_description(y)
    base64_img = _generate_time_series_plot_base64(y)
    return basic_desc, base64_img


class AutoResearchForecaster(BaseForecaster):
    """Forecaster that uses an LLM to generate and refine sktime pipeline blueprints.

    Inspired by Karpathy's autoresearch project, this forecaster:

    1. Asks an LLM to propose diverse forecasting pipeline blueprints
    2. Evaluates each blueprint on a validation split of the training data
    3. Feeds results back to the LLM for iterative refinement
    4. Selects the best-performing blueprint as the final forecaster

    Parameters
    ----------
    model : str, default="openai/gpt-4o-mini"
        LLM model identifier compatible with litellm
        (e.g., "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514").
    n_iterations : int, default=3
        Number of generate-evaluate-refine iterations.
    n_blueprints : int, default=5
        Number of blueprints to generate per iteration.
    n_fix_attempts : int, default=0
        Number of additional LLM calls to attempt fixing each failed blueprint.
        For each blueprint that fails evaluation, the LLM is asked to correct
        the spec up to ``n_fix_attempts`` times. Set to 0 to disable.
    api_params : dict or None, default=None
        Additional keyword arguments passed to litellm.completion
        (e.g., temperature, max_tokens, api_key).
    system_prompt : str or None, default=None
        Custom system prompt for the LLM. If None, uses the default prompt.
        The prompt should contain placeholders for {n_blueprints}, {forecaster_names},
        and {transformer_names} which are filled in automatically.
    refinement_prompt : str or None, default=None
        Custom refinement prompt template for the LLM. If None, uses the default prompt.
        The prompt should contain placeholders for {results_summary},
        {all_results_ranked}, {best_name}, {best_score}, and {n_blueprints} which
        are filled in automatically.
    llm_func : callable or None, default=None
        Custom callable to invoke the LLM. If None, uses litellm.completion via
        the internal ``_call_llm`` function. The callable must have the signature
        ``llm_func(messages, model, api_params) -> str``, where ``messages`` is a
        list of chat message dicts, ``model`` is the model identifier string,
        ``api_params`` is a dict of extra kwargs, and the return value is the
        raw response text. Primarily useful for testing without an API key.
    description_method : str, default="basic"
        Method for generating dataset description for the LLM. Options:

        - "basic": Text-only statistics (length, frequency, mean, std, etc.)
        - "described_plot": Generates a plot and uses a vision LLM to describe it,
          combined with basic statistics.
        - "image": Generates a plot and provides it as an image to the blueprint
          generation LLM (requires vision-capable model).

        Vision-based methods ("described_plot", "image") require a model that
        supports image input, which is checked during initialization.

    Attributes
    ----------
    best_blueprint_ : dict
        The best-performing blueprint found during fitting.
    best_score_ : float
        The validation score of the best blueprint.
    best_forecaster_ : BaseForecaster
        The fitted forecaster from the best blueprint.
    blueprint_history_ : list of dict
        History of all evaluated blueprints and their scores.
    llm_conversation_ : list of dict
        The full LLM conversation history.

    Examples
    --------
    >>> from sktime_autoresearch import AutoResearchForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import SingleWindowSplitter
    >>> y = load_airline()
    >>> forecaster = AutoResearchForecaster(  # doctest: +SKIP
    ...     cv=SingleWindowSplitter(fh=[1, 2, 3]),
    ...     model="openai/gpt-4o-mini",
    ...     n_iterations=2,
    ...     n_blueprints=3,
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["benheid"],
        "python_dependencies": ["litellm"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": True,
        "capability:missing_values": False,
        "requires-fh-in-fit": True,
        # CI and test flags
        # -----------------
        "tests:vm": True,  # tested on separate VM due to litellm dependency
        "tests:skip_by_name": ["test_doctest_examples"],
    }

    def __init__(
        self,
        cv,
        model="openai/gpt-4o-mini",
        n_iterations=3,
        n_blueprints=5,
        n_fix_attempts=0,
        api_params=None,
        system_prompt=None,
        refinement_prompt=None,
        llm_func=None,
        description_method="basic",
        estimator_info="names",
    ):
        assert estimator_info in {"names", "signatures", "docstrings"}, (
            f"estimator_info must be one of 'names', 'signatures', or 'docstrings', "
            f"got {estimator_info!r}"
        )
        # Validate description method
        assert description_method in {"basic", "described_plot", "image"}, (
            f"description_method must be one of 'basic', 'described_plot', or 'image', "
            f"got {description_method!r}"
        )
        # Warn if litellm does not recognise the model as vision-capable.
        # The check can produce false-negatives for newer/custom models, so we
        # treat it as a warning rather than a hard error.
        if description_method in {"described_plot", "image"}:
            import litellm

            if not litellm.supports_vision(model=model):
                warnings.warn(
                    f"Model '{model}' was not recognised as vision-capable by litellm. "
                    f"If the model does support vision (e.g. Gemma 4, LLaVA), "
                    f"you can ignore this warning. Otherwise switch to a vision model "
                    f"or use description_method='basic'.",
                    UserWarning,
                    stacklevel=2,
                )

        self.model = model
        self.n_iterations = n_iterations
        self.n_blueprints = n_blueprints
        self.n_fix_attempts = n_fix_attempts
        self.api_params = api_params
        self.system_prompt = system_prompt
        self.refinement_prompt = refinement_prompt
        self.cv = cv
        self.llm_func = llm_func
        self.description_method = description_method
        self.estimator_info = estimator_info

        super().__init__()

    def _generate_blueprints(self, messages):
        """Ask the LLM to generate blueprints and parse the response."""
        api_params = self.api_params or {}
        llm_func = self.llm_func if self.llm_func is not None else _call_llm
        response_text = llm_func(messages, self.model, api_params)
        messages.append({"role": "assistant", "content": response_text})
        return _parse_blueprints(response_text)

    def _build_ranked_history(self):
        """Return a formatted ranked table of all blueprints evaluated so far."""
        valid = sorted(
            (r for r in self.blueprint_history_ if r["error"] is None),
            key=lambda r: r["score"],
        )
        failed = [r for r in self.blueprint_history_ if r["error"] is not None]
        lines = [
            f"  #{i + 1} Score={r['score']:.6f}"
            + f"{r['name']}  spec={r['blueprint'].get('spec', '')}"
            for i, r in enumerate(valid)
        ] + [
            f"  FAILED  {r['name']}  spec={r['blueprint'].get('spec', '')}"
            for r in failed
        ]
        return "\n".join(lines) if lines else "  (none yet)"

    def _build_refinement_message(self, iteration_results, best_overall_result):
        """Build the refinement prompt for the next iteration."""
        refinement_prompt = (
            self.refinement_prompt
            if self.refinement_prompt is not None
            else _REFINEMENT_PROMPT
        )
        results_summary = _build_iteration_summary(iteration_results)
        all_results_ranked = self._build_ranked_history()
        return refinement_prompt.format(
            results_summary=results_summary,
            all_results_ranked=all_results_ranked,
            best_name=best_overall_result["name"]
            if best_overall_result is not None
            else "N/A",
            best_score=best_overall_result["score"]
            if best_overall_result is not None
            else float("nan"),
            n_blueprints=self.n_blueprints,
        )

    def _try_fix_blueprint(self, failed_result, messages, y, X):
        """Attempt to fix a failed blueprint via targeted LLM calls.

        Parameters
        ----------
        failed_result : dict
            Result dict from ``_evaluate_blueprint`` with a non-None error.
        messages : list of dict
            Current conversation history (used as context for the LLM).
        y, X : training data passed through to evaluation.

        Returns
        -------
        fixed_result : dict or None
            First successful result, or None if all attempts failed.
        all_attempts : list of dict
            All attempted results (successful or not).
        """
        bp = failed_result["blueprint"]
        all_attempts = []
        for attempt in range(self.n_fix_attempts):
            spec = bp.get("spec", "")
            fix_prompt = _FIX_BLUEPRINT_PROMPT.format(
                name=bp.get("name", "unnamed"),
                spec=spec,
                error=failed_result["error"],
                estimator_docs=_get_estimator_docs_from_spec(spec),
            )
            fix_messages = messages + [{"role": "user", "content": fix_prompt}]
            try:
                fixed_blueprints = self._generate_blueprints(fix_messages)
                if not fixed_blueprints:
                    continue
                fixed_bp = fixed_blueprints[0]
            except Exception as e:
                print(f"      [fix attempt {attempt + 1}] LLM call failed: {e}")
                continue

            result = _evaluate_blueprint(fixed_bp, cv=self.cv, y=y, X=X)
            all_attempts.append(result)
            self.blueprint_history_.append(result)
            status = _format_result(result, prefix="")
            print(f"      [fix attempt {attempt + 1}] {status}")
            if result["error"] is None:
                return result, all_attempts
            failed_result = result
            bp = fixed_bp
        return None, all_attempts

    def _run_iteration(self, messages, y, X, iteration):
        """Run one generate-evaluate cycle.

        Generates blueprints via the LLM, evaluates each one, prints progress,
        appends all results to ``self.blueprint_history_``, and returns the list
        of results for this iteration.

        Returns
        -------
        iteration_results : list of dict
        """
        print(f"Iteration {iteration + 1}/{self.n_iterations}")
        print("  Generating blueprints via LLM...")

        blueprints = self._generate_blueprints(messages)

        iteration_results = []
        for bp in blueprints:
            result = _evaluate_blueprint(bp, cv=self.cv, y=y, X=X)
            iteration_results.append(result)
            self.blueprint_history_.append(result)
            print(f"    {_format_result(result, prefix='')}")
            if result["error"] is not None and self.n_fix_attempts > 0:
                print(f"Trying to fix '{result['name']}', retry: {self.n_fix_attempts}")
                fixed, _ = self._try_fix_blueprint(result, messages, y, X)
                if fixed is not None:
                    iteration_results.append(fixed)

        valid_results = [r for r in iteration_results if r["error"] is None]
        if valid_results:
            best_iter = min(valid_results, key=lambda r: r["score"])
            print(
                f"  Best this iteration: {best_iter['name']} "
                f"(Score: {best_iter['score']:.6f})"
            )

        return iteration_results

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster by running the LLM blueprint search."""
        # Generate dataset description using the configured method
        if self.description_method == "basic":
            dataset_description = _get_basic_description(y)
            image_base64 = None
        elif self.description_method == "described_plot":
            dataset_description = _get_described_plot_description(
                y, self.model, self.api_params or {}
            )
            image_base64 = None
        elif self.description_method == "image":
            dataset_description, image_base64 = _get_image_description(y)
        else:
            # Should not reach here due to validation in __init__
            raise ValueError(f"Unknown description_method: {self.description_method}")

        forecaster_names, transformer_names = _build_estimator_names(
            self.estimator_info
        )

        system_prompt = (
            self.system_prompt if self.system_prompt is not None else _SYSTEM_PROMPT
        )
        system_message = {
            "role": "system",
            "content": system_prompt.format(
                n_blueprints=self.n_blueprints,
                forecaster_names=forecaster_names,
                transformer_names=transformer_names,
            ),
        }

        user_message_text = (
            f"Please propose {self.n_blueprints} diverse forecasting pipeline "
            f"blueprints for the following dataset:\n\n{dataset_description}"
        )

        if image_base64 is not None:
            user_message_content = [
                {"type": "text", "text": user_message_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        else:
            user_message_content = user_message_text

        user_message = {
            "role": "user",
            "content": user_message_content,
        }

        messages = [system_message, user_message]

        self.blueprint_history_ = []
        best_overall_score = float("inf")
        best_overall_result = None

        for iteration in range(self.n_iterations):
            try:
                iteration_results = self._run_iteration(messages, y, X, iteration)
            except Exception as e:
                print(f"  [!] LLM call failed: {e}")
                continue

            valid_results = [r for r in iteration_results if r["error"] is None]
            if valid_results:
                best_iter = min(valid_results, key=lambda r: r["score"])
                if best_iter["score"] < best_overall_score:
                    best_overall_score = best_iter["score"]
                    best_overall_result = best_iter

            if iteration < self.n_iterations - 1:
                refinement_msg = self._build_refinement_message(
                    iteration_results, best_overall_result
                )
                messages.append({"role": "user", "content": refinement_msg})

        self.llm_conversation_ = messages

        if best_overall_result is None:
            raise RuntimeError(
                "No blueprint succeeded during the search. "
                "Try increasing n_iterations or n_blueprints."
            )

        self.best_blueprint_ = best_overall_result["blueprint"]
        self.best_score_ = best_overall_score

        # Re-fit the best blueprint on full training data
        self.best_forecaster_ = craft(self.best_blueprint_["spec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_forecaster_.fit(y, X=X, fh=fh)

        return self

    def _predict(self, fh, X=None):
        """Predict using the best blueprint forecaster."""
        return self.best_forecaster_.predict(fh=fh, X=X)

    def summary(self):
        """Return a summary DataFrame of all evaluated blueprints.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with columns: name, score, error, spec.
        """
        records = [
            {
                "name": r["name"],
                "score": r["score"] if r["error"] is None else None,
                "error": r["error"],
                "spec": r["blueprint"].get("spec", ""),
            }
            for r in self.blueprint_history_
        ]
        return pd.DataFrame(records).sort_values("score", ascending=True)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict
        """
        from sktime.split import SingleWindowSplitter

        params1 = {
            "cv": SingleWindowSplitter(fh=[1]),
            "n_iterations": 1,
            "n_blueprints": 1,
            "llm_func": _mock_llm,
        }
        params2 = {
            "cv": SingleWindowSplitter(fh=[1, 2]),
            "n_iterations": 1,
            "n_blueprints": 1,
            "llm_func": _mock_llm,
        }
        return [params1, params2]
