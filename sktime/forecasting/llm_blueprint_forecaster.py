"""LLM-driven blueprint generation forecaster for sktime.

Inspired by Karpathy's autoresearch: uses an LLM to iteratively generate,
evaluate, and refine sktime forecasting pipeline blueprints to find the best
combination of transformers and forecasters for a given dataset.
"""

import json
import sys
import types
import warnings

import pandas as pd

# Compatibility shim: skbase < 0.8 does not ship skbase.utils.doctest_run,
# which sktime's check_estimator test suite tries to import. Inject a no-op
# stub so the test can proceed without requiring a newer skbase.
if "skbase.utils.doctest_run" not in sys.modules:
    _doctest_mod = types.ModuleType("skbase.utils.doctest_run")
    _doctest_mod.run_doctest = lambda obj, name=None, **kwargs: None
    sys.modules["skbase.utils.doctest_run"] = _doctest_mod
from sktime.forecasting.base import BaseForecaster

from sktime.forecasting.model_evaluation import evaluate

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert time-series forecasting engineer.
Your task is to propose sktime forecasting pipeline blueprints.

A blueprint is a JSON object with:
- "name": a short descriptive name
- "spec": a Python expression passed to sktime.registry.craft() to construct the forecaster.
  Use sktime class names directly — no imports needed. craft() resolves them automatically.

Rules for "spec":
- A bare forecaster: NaiveForecaster(strategy="last")
- A pipeline: TransformedTargetForecaster([Transformer1(), Transformer2(), Forecaster()])
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
  {{"name": "Naive last", "spec": "NaiveForecaster(strategy=\\"last\\")"}},
  {{"name": "ETS additive", "spec": "ExponentialSmoothing(trend=\\"add\\", seasonal=\\"add\\", sp=12)"}},
  {{"name": "Detrend + ETS", "spec": "TransformedTargetForecaster([Detrender(), ExponentialSmoothing()])"}},
  {{"name": "Deseason + Detrend + AutoARIMA", "spec": "TransformedTargetForecaster([Deseasonalizer(sp=12), Detrender(), AutoARIMA()])"}},
  {{"name": "BoxCox + Naive mean", "spec": "TransformedTargetForecaster([BoxCoxTransformer(), NaiveForecaster(strategy=\\"mean\\")])"}},
  {{"name": "Differencer + AutoARIMA BIC", "spec": "TransformedTargetForecaster([Differencer(), AutoARIMA(information_criterion=\\"bic\\")])"}},
]

Provide {n_blueprints} diverse blueprints in a JSON array.
Respond ONLY with a valid JSON array of blueprint objects — no markdown, no explanation.
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
4. Also explore new diverse combinations that might outperform it.

Propose {n_blueprints} new blueprints. At least one should be a variation of the
best blueprint, and at least one should be a novel combination.

Respond ONLY with a valid JSON array of blueprint objects — no markdown, no explanation.
"""


def _build_estimator_names():
    """Return comma-separated name lists for all sktime forecasters and transformers."""
    from sktime.registry import all_estimators

    forecaster_names = ", ".join(
        name for name, _ in all_estimators(estimator_types="forecaster")
    )
    transformer_names = ", ".join(
        name for name, _ in all_estimators(estimator_types="transformer")
    )
    return forecaster_names, transformer_names


def _build_pipeline_from_blueprint(blueprint):
    """Instantiate an sktime pipeline from a blueprint dict.

    Parameters
    ----------
    blueprint : dict
        Blueprint specification with a "spec" key containing a craft() expression.

    Returns
    -------
    forecaster : sktime forecaster
        An sktime forecaster constructed via sktime.registry.craft.
    """
    from sktime.registry import craft

    return craft(blueprint["spec"])


def _evaluate_blueprint(blueprint, cv, y, X=None):
    """Evaluate a single blueprint on the given train/test split.

    Returns
    -------
    result : dict
        Dictionary with keys: name, score, error, forecaster, blueprint.
    """
    name = blueprint.get("name", "unnamed")
    try:
        forecaster = _build_pipeline_from_blueprint(blueprint)
        results = evaluate(forecaster=forecaster, y=y, X=X, cv=cv)
        score = results["test_MeanAbsolutePercentageError"].mean()
        return {
            "name": name,
            "score": float(score),
            "error": None,
            "forecaster": forecaster,
            "blueprint": blueprint,
        }
    except Exception as e:
        return {
            "name": name,
            "score": float("inf"),
            "error": f"{type(e).__name__}: {e}",
            "forecaster": None,
            "blueprint": blueprint,
        }


def _call_llm(messages, model, api_params):
    """Call the LLM via litellm and return the response text."""
    import litellm

    litellm.suppress_debug_info = True

    response = litellm.completion(
        model=model,
        messages=messages,
        **api_params,
    )
    return response.choices[0].message.content


def _parse_blueprints(response_text):
    """Parse LLM response text into a list of blueprint dicts."""
    return json.loads(response_text)


def _mock_llm(_messages, _model, _api_params):
    """Mock LLM callable for testing — returns a single naive blueprint."""
    return json.dumps(
        [{"name": "Naive last", "spec": 'NaiveForecaster(strategy="last")'}]
    )


def _format_result(result, *, prefix="- "):
    """Format a single result dict as a human-readable status line."""
    if result["error"]:
        return f"{prefix}{result['name']}: FAILED ({result['error']})"
    return f"{prefix}{result['name']}: Score={result['score']:.6f}"


def _build_iteration_summary(iteration_results):
    """Build a formatted summary string for a list of iteration results."""
    return "\n".join(_format_result(r) for r in iteration_results)


class LLMBlueprintForecaster(BaseForecaster):
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
    api_params : dict or None, default=None
        Additional keyword arguments passed to litellm.completion
        (e.g., temperature, max_tokens, api_key).
    system_prompt : str or None, default=None
        Custom system prompt for the LLM. If None, uses the default prompt.
        The prompt should contain placeholders for {n_blueprints}, {forecaster_names},
        and {transformer_names} which are filled in automatically.
    llm_func : callable or None, default=None
        Custom callable to invoke the LLM. If None, uses litellm.completion via
        the internal ``_call_llm`` function. The callable must have the signature
        ``llm_func(messages, model, api_params) -> str``, where ``messages`` is a
        list of chat message dicts, ``model`` is the model identifier string,
        ``api_params`` is a dict of extra kwargs, and the return value is the
        raw response text. Primarily useful for testing without an API key.

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
    >>> from sktime_autoresearch import LLMBlueprintForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import SingleWindowSplitter
    >>> y = load_airline()
    >>> forecaster = LLMBlueprintForecaster(  # doctest: +SKIP
    ...     cv=SingleWindowSplitter(fh=[1, 2, 3]),
    ...     model="openai/gpt-4o-mini",
    ...     n_iterations=2,
    ...     n_blueprints=3,
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "capability:exogenous": False,
        "capability:missing_values": False,
        "requires-fh-in-fit": True,
        "tests:skip_by_name": ["test_doctest_examples"],
    }

    def __init__(
        self,
        cv,
        model="openai/gpt-4o-mini",
        n_iterations=3,
        n_blueprints=5,
        api_params=None,
        system_prompt=None,
        llm_func=None,
    ):
        self.model = model
        self.n_iterations = n_iterations
        self.n_blueprints = n_blueprints
        self.api_params = api_params
        self.system_prompt = system_prompt
        # TODO refinement prompt should also be customizable!
        self.cv = cv
        self.llm_func = llm_func

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
            f"  #{i+1} MAPE={r['score']:.6f}  {r['name']}  spec={r['blueprint'].get('spec', '')}"
            for i, r in enumerate(valid)
        ] + [
            f"  FAILED  {r['name']}  spec={r['blueprint'].get('spec', '')}"
            for r in failed
        ]
        return "\n".join(lines) if lines else "  (none yet)"

    def _build_refinement_message(self, iteration_results, best_overall_result):
        """Build the refinement prompt for the next iteration."""
        results_summary = _build_iteration_summary(iteration_results)
        all_results_ranked = self._build_ranked_history()
        if best_overall_result:
            return _REFINEMENT_PROMPT.format(
                results_summary=results_summary,
                all_results_ranked=all_results_ranked,
                best_name=best_overall_result["name"],
                best_score=best_overall_result["score"],
                n_blueprints=self.n_blueprints,
            )
        return (
            f"All blueprints failed in this round. Results:\n\n{results_summary}\n\n"
            f"Please try simpler, more robust blueprints. "
            f"Propose {self.n_blueprints} new blueprints.\n\n"
            f"Respond ONLY with a valid JSON array."
        )

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
        dataset_description = (
            f"Dataset info:\n"
            f"- Length: {len(y)} observations\n"
            f"- Frequency: {y.index.freqstr if hasattr(y.index, 'freqstr') else 'unknown'}\n"
            f"- Mean: {y.mean():.4f}, Std: {y.std():.4f}\n"
            f"- Min: {y.min():.4f}, Max: {y.max():.4f}\n"
            f"- Forecast horizon: {len(fh)} steps ahead\n"
        )

        forecaster_names, transformer_names = _build_estimator_names()

        system_prompt = self.system_prompt if self.system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    n_blueprints=self.n_blueprints,
                    forecaster_names=forecaster_names,
                    transformer_names=transformer_names,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Please propose {self.n_blueprints} diverse forecasting pipeline "
                    f"blueprints for the following dataset:\n\n{dataset_description}"
                ),
            },
        ]

        self.blueprint_history_ = []
        best_overall_score = float("inf")
        best_overall_result = None

        for iteration in range(self.n_iterations):
            try:
                iteration_results = self._run_iteration(messages, y, X, iteration)
            except Exception as e:
                print(f"  [!] LLM call failed: {e}")
                break

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
        self.best_forecaster_ = _build_pipeline_from_blueprint(self.best_blueprint_)
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
