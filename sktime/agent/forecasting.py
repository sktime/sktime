# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasting agent for sktime."""

import pandas as pd
from sktime.agent.base import BaseAgent
from sktime.registry import all_estimators

class ForecastingAgent(BaseAgent):
    """Agent for generating forecasting pipelines.

    Parameters
    ----------
    llm : BaseLLM
        The LLM interface to use for code generation.
    """

    def __init__(self, llm):
        self.llm = llm
        super().__init__()

    def _get_forecasters_info(self):
        """Fetch available forecasters and their tags."""
        forecasters = all_estimators(
            estimator_types="forecaster",
            as_dataframe=True,
            return_tags=["capability:multivariate", "capability:exogenous", "capability:missing_values"]
        )
        # Simplify the info for the prompt
        info = []
        for _, row in forecasters.iterrows():
            name = row["name"]
            tags = ", ".join([f"{k}={v}" for k, v in row.items() if k.startswith("capability")])
            info.append(f"- {name} ({tags})")
        return "\n".join(info)

    def generate_pipeline(self, query: str):
        """Generate a forecasting pipeline from a natural language query.

        Parameters
        ----------
        query : str
            The natural language query.

        Returns
        -------
        BaseForecaster
            The generated sktime forecasting pipeline.
        """
        forecasters_info = self._get_forecasters_info()
        
        prompt = f"""
You are an expert in sktime, a library for time series analysis.
The user wants to: "{query}"

Available forecasters in sktime and their capabilities:
{forecasters_info}

Common transformations in sktime:
- TabularToSeriesAdaptor(StandardScaler())
- Deseasonalizer(sp=12)
- Differencer()

Based on the user's intent, generate a Python code snippet that defines a variable named 'forecaster'.
The 'forecaster' should be an sktime object (e.g., a simple forecaster or a ForecastingPipeline).
ONLY return the Python code, no explanation.

Example:
from sktime.forecasting.arima import ARIMA
forecaster = ARIMA()
"""
        code = self.llm.generate_code(prompt)
        
        # Execute the code in a local namespace
        local_ns = {}
        # We need to provide some common imports if the LLM doesn't include them,
        # but the prompt asks the LLM to include them.
        try:
            exec(code, {}, local_ns)
        except Exception as e:
            raise RuntimeError(f"Failed to execute generated code: {e}\nCode:\n{code}")
            
        if "forecaster" not in local_ns:
            raise RuntimeError(f"Generated code did not define 'forecaster'.\nCode:\n{code}")
            
        return local_ns["forecaster"]

    def ask(self, query: str):
        """Alias for generate_pipeline for BaseAgent compatibility."""
        return self.generate_pipeline(query)
