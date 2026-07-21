"""Core agent logic for parsing user intent and running sktime tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

from utils import compute_basic_stats, compute_trend, detect_anomaly_hints


@dataclass
class Intent:
    task: str
    horizon: int | None = None
    analysis_type: str | None = None


class MockLLM:

    def parse(self, query: str) -> dict[str, Any]:
        q = query.lower().strip()

        if "forecast" in q or "predict" in q:
            match = re.search(r"(next\s+)?(\d+)\s*(step|steps|point|points|period|periods)", q)
            horizon = int(match.group(2)) if match else 12
            return {"task": "forecast", "horizon": horizon}

        if "trend" in q:
            return {"task": "analysis", "analysis_type": "trend"}

        if "mean" in q or "average" in q or "summary" in q or "stats" in q:
            return {"task": "analysis", "analysis_type": "mean"}

        if "anomal" in q or "outlier" in q or "spike" in q:
            return {"task": "analysis", "analysis_type": "anomaly"}

        return {"task": "analysis", "analysis_type": "mean"}

    def explain(self, intent: Intent, payload: dict[str, Any]) -> str:
        if intent.task == "forecast":
            horizon = intent.horizon or len(payload.get("forecast", []))
            return (
                f"I forecasted the next {horizon} steps using a NaiveForecaster (last-value strategy). "
                "The values below are point forecasts."
            )

        if intent.analysis_type == "trend":
            return "I ran a simple linear trend analysis on the full series."

        if intent.analysis_type == "anomaly":
            return "I checked for anomaly hints using z-score thresholding."

        return "I computed basic descriptive statistics for the series."


def parse_query(query: str, llm: MockLLM | None = None) -> dict[str, Any]:
    llm = llm or MockLLM()
    parsed = llm.parse(query)
    return parsed


def _to_intent(parsed: dict[str, Any]) -> Intent:
    return Intent(
        task=str(parsed.get("task", "analysis")),
        horizon=parsed.get("horizon"),
        analysis_type=parsed.get("analysis_type"),
    )


def _run_forecast(series: pd.Series, horizon: int) -> dict[str, Any]:
    # Normalize to integer index to avoid PeriodIndex/offset frequency conflicts.
    y = series.copy()
    y.index = pd.RangeIndex(start=0, stop=len(y), step=1)

    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(y)

    fh = np.arange(1, horizon + 1)
    pred = forecaster.predict(fh=fh)

    return {
        "model": "NaiveForecaster(strategy='last')",
        "forecast": {str(idx): float(value) for idx, value in pred.items()},
    }


def _run_analysis(series: pd.Series, analysis_type: str | None) -> dict[str, Any]:
    analysis_type = (analysis_type or "mean").lower()

    if analysis_type == "trend":
        trend = compute_trend(series)
        return {"analysis_type": "trend", "trend": trend}

    if analysis_type == "anomaly":
        anomalies = detect_anomaly_hints(series)
        return {"analysis_type": "anomaly", "anomalies": anomalies}

    stats = compute_basic_stats(series)
    return {"analysis_type": "mean", "stats": stats}


def run_agent(query: str, data: pd.Series, llm: MockLLM | None = None) -> str:
    llm = llm or MockLLM()

    parsed = parse_query(query, llm=llm)
    intent = _to_intent(parsed)

    if intent.task == "forecast":
        horizon = int(intent.horizon or 12)
        result = _run_forecast(data, horizon=horizon)
    elif intent.task == "analysis":
        result = _run_analysis(data, analysis_type=intent.analysis_type)
    else:
        result = {
            "analysis_type": "mean",
            "stats": compute_basic_stats(data),
            "note": f"Unknown task '{intent.task}', defaulted to basic analysis.",
        }

    narrative = llm.explain(intent, result)
    body = json.dumps(result, indent=2)
    return f"Query: {query}\nIntent: {parsed}\n\n{narrative}\n\nResult:\n{body}"
